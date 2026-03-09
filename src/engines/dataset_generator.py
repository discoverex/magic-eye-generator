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
        asset_id = asset['id']
        asset_display_name = asset['display_name']

        asset_output_dir = os.path.join(self.dataset_base_dir, asset_id)
        # 💡 주의: 배치 사이즈가 1보다 크면 파일명 규칙을 변경해야 할 수도 있습니다.
        # 여기서는 현재 구조를 유지하면서 리스트의 첫 번째 결과만 저장하는 방식으로 수정합니다.
        prob_path = os.path.join(asset_output_dir, f"{asset_id}_{index}_prob.png")
        ans_path = os.path.join(asset_output_dir, f"{asset_id}_{index}_ans.png")

        if os.path.exists(prob_path) and os.path.exists(ans_path):
            pbar.update(1)
            return

        async with self.semaphore:
            try:
                llm_prompts = await self.service.prompt_agent.get_diverse_prompts(asset_display_name, count=1)
                current_prompt = llm_prompts[0] if llm_prompts else asset['prompt']

                # 💡 서비스 호출 (이제 결과가 리스트로 옵니다)
                generated_results = await self.service.generate_specific_game(
                    asset,
                    current_prompt,
                    step_callback=None,
                    batch_size=8  # Out of Memory 나기 직전까지 두배씩 올려보기
                )

                # 💡 리스트에서 첫 번째 결과물을 꺼내서 처리
                if not generated_results:
                    raise Exception("생성된 이미지가 없습니다.")

                generated_data = generated_results[0]

                # 파일 저장
                with open(prob_path, "wb") as f:
                    f.write(generated_data.problem_image)
                with open(ans_path, "wb") as f:
                    f.write(generated_data.answer_image)

                # 메타데이터 기록
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