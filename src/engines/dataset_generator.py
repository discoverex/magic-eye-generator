import asyncio
import csv
import os
import random
from datetime import datetime
from typing import Optional

from tqdm import tqdm

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
        
        # 전체 진행바 설정 (단일 바 통합)
        main_pbar = tqdm(total=total_expected, desc="생성 준비 중...", unit="장", position=0, leave=True)

        for a_idx, asset in enumerate(MAGIC_EYE_ASSETS, 1):
            asset_id = asset['id']
            asset_display_name = asset['display_name']

            # 에셋별 폴더 생성
            asset_output_dir = os.path.join(self.dataset_base_dir, asset_id)
            os.makedirs(asset_output_dir, exist_ok=True)

            for i in range(self.num_images_per_asset):
                current_info = f"{asset_display_name} ({i+1}/{self.num_images_per_asset})"
                main_pbar.set_description(f"{current_info} [대기]")

                prob_filename = f"{asset_id}_{i + 1}_prob.png"
                ans_filename = f"{asset_id}_{i + 1}_ans.png"
                prob_path = os.path.join(asset_output_dir, prob_filename)
                ans_path = os.path.join(asset_output_dir, ans_filename)

                # 중복 체크
                if os.path.exists(prob_path) and os.path.exists(ans_path):
                    main_pbar.update(1)
                    continue

                try:
                    # 미니 바 생성 함수
                    def get_mini_bar(step, total):
                        width = 8
                        filled = int(width * step / total)
                        return "|" + "█" * filled + " " * (width - filled) + "|"

                    # 1. LLM을 통한 다양한 프롬프트 확보
                    main_pbar.set_description(f"{asset_display_name} ({i+1}) [프롬프트]")
                    llm_prompts = await self.service.prompt_agent.get_diverse_prompts(asset_display_name, count=1)
                    current_prompt = llm_prompts[0] if llm_prompts else asset['prompt']

                    # 2. 이미지 생성 실행 (미니 바 콜백 적용)
                    def step_update(step, total_steps):
                        mbar = get_mini_bar(step + 1, total_steps)
                        main_pbar.set_description(f"{asset_display_name} ({i+1}) {mbar}")

                    generated_data = await self.service.generate_specific_game(
                        asset, 
                        current_prompt, 
                        step_callback=step_update
                    )

                    # 3. 파일 저장 및 메타데이터 기록
                    main_pbar.set_description(f"{asset_display_name} ({i+1}) [저장 중]")
                    with open(prob_path, "wb") as f:
                        f.write(generated_data.problem_image)
                    with open(ans_path, "wb") as f:
                        f.write(generated_data.answer_image)

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

                    main_pbar.update(1)

                except Exception as e:
                    tqdm.write(f"    ❌ {asset_id} 생성 중 오류 발생: {str(e)}")
                    await asyncio.sleep(2)
        
        main_pbar.set_description("생성 완료!")
        main_pbar.close()
        # 하단 진행바가 종료된 후 터미널 커서를 정리하기 위해 줄 바꿈 추가
        print("\n" * 2) 
        print("✨ 데이터셋 구축 완료! 모든 이미지가 'datasets' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    import sys
    
    # 명령행 인자로 개수를 전달받으면 해당 값을 사용하고, 없으면 기본값 300 사용
    num_images = 300
    if len(sys.argv) > 1:
        try:
            num_images = int(sys.argv[1])
        except ValueError:
            print(f"⚠️ 경고: '{sys.argv[1]}'은 올바른 숫자가 아닙니다. 기본값 {num_images}으로 진행합니다.")

    generator = DatasetGenerator(num_images_per_asset=num_images)
    asyncio.run(generator.run())
