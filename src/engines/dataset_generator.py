import asyncio
import csv
import os
import random
from datetime import datetime
from typing import Optional

from tqdm.asyncio import tqdm  # 비동기 환경에 최적화된 tqdm

from src.config.settings import BASE_DIR
from src.consts.magic_eye_assets import MAGIC_EYE_ASSETS
from src.services.magic_eye_service import MagicEyeService


class DatasetGenerator:
    """
    매직아이 이미지와 정답(Depth Map)을 비동기 배치 방식으로 생성하여 저장합니다.
    Quadro RTX 4000 (8GB) 환경을 고려하여 동시 실행 수를 제한합니다.
    """

    def __init__(self, num_images_per_asset: int = 300, max_concurrent: int = 1):
        self.num_images_per_asset = num_images_per_asset
        # 💡 VRAM 한계를 고려하여 세마포어 설정 (8GB 환경에서는 1 권장)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.dataset_base_dir = os.path.join(BASE_DIR, "datasets")
        self.metadata_path = os.path.join(self.dataset_base_dir, "metadata.csv")
        self.service: Optional[MagicEyeService] = None

    def _prepare_directories(self):
        os.makedirs(self.dataset_base_dir, exist_ok=True)

    def _init_metadata_csv(self):
        if not os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'asset_id', 'display_name', 'description',
                    'problem_path', 'answer_path', 'split', 'created_at'
                ])

    async def _init_service(self):
        print("🔄 모델 로딩 중... 메모리 점유율을 확인하세요.")
        self.service = MagicEyeService()

    def _get_split_type(self) -> str:
        rand_val = random.random()
        if rand_val < 0.8:
            return 'train'
        elif rand_val < 0.9:
            return 'val'
        return 'test'

    async def _generate_task(self, asset: dict, index: int, pbar: tqdm):
        """
        단일 이미지 생성 태스크: 세마포어를 통해 GPU 접근을 제어합니다.
        """
        asset_id = asset['id']
        asset_display_name = asset['display_name']

        asset_output_dir = os.path.join(self.dataset_base_dir, asset_id)
        prob_path = os.path.join(asset_output_dir, f"{asset_id}_{index}_prob.png")
        ans_path = os.path.join(asset_output_dir, f"{asset_id}_{index}_ans.png")

        # 이미 존재하는 파일 스킵
        if os.path.exists(prob_path) and os.path.exists(ans_path):
            pbar.update(1)
            return

        # 💡 GPU 자원 획득 대기
        async with self.semaphore:
            try:
                # 1. 프롬프트 생성 (이 단계는 GPU를 덜 쓰므로 세마포어 밖으로 빼도 됩니다)
                llm_prompts = await self.service.prompt_agent.get_diverse_prompts(asset_display_name, count=1)
                current_prompt = llm_prompts[0] if llm_prompts else asset['prompt']

                # 2. 이미지 생성
                # 세마포어 안에서 실행되어 한 번에 설정된 개수만큼만 GPU를 점유합니다.
                generated_data = await self.service.generate_specific_game(
                    asset,
                    current_prompt,
                    step_callback=None  # 비동기 배치에서는 콜백 출력이 겹치므로 비활성
                )

                # 3. 파일 저장 및 메타데이터 기록 (I/O)
                with open(prob_path, "wb") as f:
                    f.write(generated_data.problem_image)
                with open(ans_path, "wb") as f:
                    f.write(generated_data.answer_image)

                split = self._get_split_type()
                with open(self.metadata_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        asset_id, asset_display_name, current_prompt,
                        os.path.relpath(prob_path, self.dataset_base_dir),
                        os.path.relpath(ans_path, self.dataset_base_dir),
                        split, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ])

            except Exception as e:
                tqdm.write(f"    ❌ {asset_id} ({index}) 생성 중 오류: {str(e)}")
            finally:
                pbar.update(1)

    async def run(self):
        self._prepare_directories()
        self._init_metadata_csv()
        await self._init_service()

        total_assets = len(MAGIC_EYE_ASSETS)
        total_expected = total_assets * self.num_images_per_asset

        # 전체 진행 상황을 관리할 비동기 pbar
        main_pbar = tqdm(total=total_expected, desc="🚀 데이터셋 생성 중", unit="장")

        # 모든 작업 예약
        tasks = []
        for asset in MAGIC_EYE_ASSETS:
            # 에셋별 폴더 미리 생성
            os.makedirs(os.path.join(self.dataset_base_dir, asset['id']), exist_ok=True)
            for i in range(1, self.num_images_per_asset + 1):
                tasks.append(self._generate_task(asset, i, main_pbar))

        # 💡 동시에 실행 (세마포어가 내부적으로 개수 조절)
        await asyncio.gather(*tasks)

        main_pbar.close()
        print("\n✨ 모든 작업이 완료되었습니다!")


if __name__ == "__main__":
    import sys

    num_images = 300
    if len(sys.argv) > 1:
        try:
            num_images = int(sys.argv[1])
        except ValueError:
            pass

    # Quadro RTX 4000 (8GB) 환경이므로 max_concurrent=1 추천
    # 만약 VRAM이 더 비어있다면 2로 올려보세요.
    generator = DatasetGenerator(num_images_per_asset=num_images, max_concurrent=1)
    asyncio.run(generator.run())