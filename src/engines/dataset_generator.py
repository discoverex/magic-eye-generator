import asyncio
import csv
import multiprocessing
import os
import random
import re
import signal
from datetime import datetime
from typing import Optional, Set, Tuple

import numpy as np
from PIL import Image
from tqdm.asyncio import tqdm

from src.config.settings import BASE_DIR
from src.consts.magic_eye_assets import MAGIC_EYE_ASSETS
from src.services.magic_eye_service import MagicEyeService
from src.utils.stereogram import create_stereogram
from src.utils.split_helper import get_split_from_index


def _cpu_worker_process(input_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
    """별도 프로세스에서 CPU 연산(스테레오그램) 수행"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while True:
        task = input_queue.get()
        if task is None: break
        depth_array, asset_dict, idx, prompt, prob_path, ans_path = task
        try:
            depth_map_pil = Image.fromarray(depth_array)
            magic_eye_img = create_stereogram(depth_map_pil)
            magic_eye_img.save(prob_path, "PNG")
            depth_map_pil.save(ans_path, "PNG")
            result_queue.put((asset_dict, idx, prompt, prob_path, ans_path))
        except Exception as e:
            print(f"Worker Error: {e}")


class DatasetGenerator:
    """
    최종 진화형 데이터셋 생성기.
    프롬프트 프리페칭 큐와 멀티프로세싱을 결합하여 GPU 버블을 최소화합니다.
    """

    def __init__(self, num_images_per_asset: int = 300, batch_size: int = 2):
        self.num_images_per_asset = num_images_per_asset
        self.batch_size = batch_size
        
        # 💡 리소스 제어
        self.gpu_semaphore = asyncio.Semaphore(1)
        self.file_lock = asyncio.Lock()
        
        # 💡 프롬프트 프리페칭을 위한 전용 비동기 큐 (지연 시간 0 목표)
        self.prompt_queue = asyncio.Queue(maxsize=100)
        
        # 💡 멀티프로세싱 자원
        self.manager = multiprocessing.Manager()
        self.input_queue = self.manager.Queue(maxsize=50)
        self.result_queue = self.manager.Queue()
        self.workers = []
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)
        
        self.dataset_base_dir = os.path.join(BASE_DIR, "datasets")
        self.metadata_path = os.path.join(self.dataset_base_dir, "metadata.csv")
        self.service: Optional[MagicEyeService] = None

    async def _prompt_prefetch_worker(self):
        """백그라운드에서 OpenAI API를 미리 호출하여 큐를 채웁니다."""
        print("🚀 프롬프트 프리페칭 워커 가동...")
        while not hasattr(self, '_stop_fetching') or not self._stop_fetching:
            for asset in MAGIC_EYE_ASSETS:
                if self.prompt_queue.full():
                    await asyncio.sleep(1)
                    continue
                try:
                    # 에셋당 1개씩 순회하며 미리 프롬프트 생성
                    prompts = await self.service.prompt_agent.get_diverse_prompts(asset['display_name'], count=1)
                    await self.prompt_queue.put((asset, prompts[0]))
                except:
                    await asyncio.sleep(2)

    async def _metadata_writer_loop(self, pbar: tqdm):
        """완료된 작업을 기록하고 진행바를 업데이트합니다."""
        while True:
            try:
                result = self.result_queue.get_nowait()
                if result is None: break
                
                asset, idx, prompt, prob_path, ans_path = result
                split = get_split_from_index(idx, self.num_images_per_asset)
                
                async with self.file_lock:
                    with open(self.metadata_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            asset['id'], asset['display_name'], prompt,
                            os.path.relpath(prob_path, self.dataset_base_dir),
                            os.path.relpath(ans_path, self.dataset_base_dir),
                            split, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ])
                pbar.update(1)
            except:
                await asyncio.sleep(0.1)
                if hasattr(self, '_stop_writer') and self._stop_writer and self.result_queue.empty():
                    break

    async def _generate_batch_task(self, indices: list[int]):
        """GPU는 오직 생성에만 집중합니다 (프롬프트는 큐에서 즉시 획득)"""
        try:
            # 1. 프롬프트 즉시 획득 (대기 시간 거의 없음)
            asset, prompt = await self.prompt_queue.get()

            # 2. GPU 추론
            async with self.gpu_semaphore:
                raw_outputs = await self.service.generate_raw_outputs(prompt, batch_size=len(indices))
            
            if not raw_outputs: return

            # 3. CPU 큐로 데이터 전송 (NumPy 변환 및 삽입)
            # 이 과정도 비동기 태스크로 던져서 GPU 세마포어 반납을 앞당길 수 있음
            for i, (raw_image, depth_map) in enumerate(raw_outputs):
                idx = indices[i]
                asset_dir = os.path.join(self.dataset_base_dir, asset['id'])
                p_path = os.path.join(asset_dir, f"{asset['id']}_{idx}_prob.png")
                a_path = os.path.join(asset_dir, f"{asset['id']}_{idx}_ans.png")
                
                # 피클링 오버헤드 최소화를 위해 NumPy 변환
                depth_arr = np.array(depth_map)
                self.input_queue.put((depth_arr, asset, idx, prompt, p_path, a_path))

        except Exception as e:
            print(f"❌ GPU Task Error: {e}")

    async def run(self):
        os.makedirs(self.dataset_base_dir, exist_ok=True)
        self._init_metadata_csv()
        completed = self._get_completed_items()
        
        # 워커 및 서비스 시작
        for _ in range(self.num_workers):
            p = multiprocessing.Process(target=_cpu_worker_process, args=(self.input_queue, self.result_queue))
            p.start()
            self.workers.append(p)
            
        self.service = MagicEyeService()
        
        # 💡 프롬프트 프리페칭 시작
        self._stop_fetching = False
        prefetch_task = asyncio.create_task(self._prompt_prefetch_worker())

        # 작업 대기열 생성
        tasks_to_do = []
        for asset in MAGIC_EYE_ASSETS:
            os.makedirs(os.path.join(self.dataset_base_dir, asset['id']), exist_ok=True)
            indices = [i for i in range(1, self.num_images_per_asset + 1) if (asset['id'], i) not in completed]
            for i in range(0, len(indices), self.batch_size):
                tasks_to_do.append(indices[i:i + self.batch_size])

        if not tasks_to_do:
            print("✅ 추가 생성할 데이터가 없습니다.")
            self._cleanup()
            return

        pbar = tqdm(total=len(tasks_to_do) * self.batch_size + len(completed), desc="🚀")
        pbar.n = len(completed)
        pbar.refresh()

        self._stop_writer = False
        writer_task = asyncio.create_task(self._metadata_writer_loop(pbar))

        # 💡 GPU 생산 실행
        # 모든 작업을 한꺼번에 예약 (세마포어가 GPU 사용량 조절)
        gpu_tasks = [self._generate_batch_task(batch) for batch in tasks_to_do]
        await asyncio.gather(*gpu_tasks)

        # 종료 처리
        self._stop_fetching = True
        prefetch_task.cancel()
        self._stop_writer = True
        await writer_task
        
        for _ in range(self.num_workers): self.input_queue.put(None)
        for p in self.workers: p.join()
        pbar.close()

    def _init_metadata_csv(self):
        if not os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['asset_id', 'display_name', 'description', 'problem_path', 'answer_path', 'split', 'created_at'])

    def _get_completed_items(self) -> Set[Tuple[str, int]]:
        completed = set()
        if not os.path.exists(self.metadata_path): return completed
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                match = re.search(r'_(\d+)_prob\.png', row.get('problem_path', ''))
                if match: completed.add((row['asset_id'], int(match.group(1))))
        return completed

    def _cleanup(self):
        # 자원 정리 로직 (생략 가능)
        pass


if __name__ == "__main__":
    import sys
    try: multiprocessing.set_start_method('spawn', force=True)
    except: pass
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    generator = DatasetGenerator(num_images_per_asset=num, batch_size=2)
    asyncio.run(generator.run())
