import asyncio
import csv
import os
import random
from datetime import datetime
from typing import Optional

from src.config.settings import BASE_DIR
from src.consts.magic_eye_assets import MAGIC_EYE_ASSETS
from src.services.magic_eye_service import MagicEyeService


class DatasetGenerator:
    """
    매직아이 이미지와 정답(Depth Map)을 생성하여
    학습용 데이터셋(CSV 메타데이터 포함) 구조로 저장하는 클래스
    """

    def __init__(self, num_images_per_asset: int = 300):
        self.num_images_per_asset = num_images_per_asset
        self.dataset_base_dir = os.path.join(BASE_DIR, "datasets")
        self.metadata_path = os.path.join(self.dataset_base_dir, "metadata.csv")
        self.service: Optional[MagicEyeService] = None

    def _prepare_directories(self):
        """
        데이터셋 저장을 위한 기본 디렉토리를 생성합니다.
        """
        os.makedirs(self.dataset_base_dir, exist_ok=True)

    def _init_metadata_csv(self):
        """
        메타데이터 CSV 파일이 없으면 헤더를 생성합니다.
        """
        if not os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'asset_id',  # 사물 ID (라벨링용)
                    'display_name',  # 사물 이름
                    'description',  # LLM이 생성한 상세 묘사 (데이터 특징 기록)
                    'problem_path',  # 입력 이미지 경로 (X)
                    'answer_path',  # 정답 이미지 경로 (Y)
                    'split',  # train / val / test 분할
                    'created_at'  # 생성 일시
                ])

    async def _init_service(self):
        """
        MagicEyeService를 초기화합니다. (모델 로딩 포함)
        """
        print("🔄 모델 로딩 중... 모델이 메모리에 올라갈 때까지 잠시만 기다려주세요.")
        self.service = MagicEyeService()

    def _get_split_type(self) -> str:
        """
        데이터셋 분할 타입을 결정합니다 (80% Train, 10% Val, 10% Test).
        """
        rand_val = random.random()
        if rand_val < 0.8:
            return 'train'
        elif rand_val < 0.9:
            return 'val'
        return 'test'

    async def run(self):
        """
        데이터셋 생성 프로세스 전체를 실행합니다.
        """
        self._prepare_directories()
        self._init_metadata_csv()
        await self._init_service()

        total_assets = len(MAGIC_EYE_ASSETS)
        total_expected = total_assets * self.num_images_per_asset
        completed_count = 0

        print(f"🚀 총 {total_assets}개의 에셋에 대해 각각 {self.num_images_per_asset}장씩 생성을 시작합니다.")

        for a_idx, asset in enumerate(MAGIC_EYE_ASSETS, 1):
            asset_id = asset['id']
            asset_display_name = asset['display_name']

            # 에셋별 폴더 생성
            asset_output_dir = os.path.join(self.dataset_base_dir, asset_id)
            os.makedirs(asset_output_dir, exist_ok=True)

            print(f"\n📦 [{a_idx}/{total_assets}] 카테고리: {asset_display_name} ({asset_id})")

            for i in range(self.num_images_per_asset):
                prob_filename = f"{asset_id}_{i + 1}_prob.png"
                ans_filename = f"{asset_id}_{i + 1}_ans.png"
                prob_path = os.path.join(asset_output_dir, prob_filename)
                ans_path = os.path.join(asset_output_dir, ans_filename)

                # 중복 체크
                if os.path.exists(prob_path) and os.path.exists(ans_path):
                    completed_count += 1
                    continue

                start_time = datetime.now()
                print(f"  [Image {i + 1}/{self.num_images_per_asset}] 생성 중...")

                try:
                    # 1. LLM을 통한 다양한 프롬프트 확보
                    llm_prompts = await self.service.prompt_agent.get_diverse_prompts(asset_display_name, count=1)
                    current_prompt = llm_prompts[0] if llm_prompts else asset['prompt']

                    # 2. 이미지 생성 실행
                    generated_data = await self.service.generate_specific_game(asset, current_prompt)

                    # 3. 파일 저장
                    with open(prob_path, "wb") as f:
                        f.write(generated_data.problem_image)
                    with open(ans_path, "wb") as f:
                        f.write(generated_data.answer_image)

                    # 4. 메타데이터 기록
                    split = self._get_split_type()
                    with open(self.metadata_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            asset_id,
                            asset_display_name,
                            current_prompt,
                            os.path.relpath(prob_path, self.dataset_base_dir),
                            os.path.relpath(ans_path, self.dataset_base_dir),
                            split,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ])

                    duration = (datetime.now() - start_time).seconds
                    completed_count += 1
                    progress = (completed_count / total_expected) * 100
                    print(f"    ✅ 완료! ({duration}s) | 분할: {split.upper()} | 진행률: {progress:.1f}%")

                except Exception as e:
                    print(f"    ❌ {asset_id} 생성 중 오류 발생: {str(e)}")
                    await asyncio.sleep(2)

        print("\n✨ 데이터셋 구축 완료! 모든 이미지가 'datasets' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    generator = DatasetGenerator(num_images_per_asset=300)
    asyncio.run(generator.run())
