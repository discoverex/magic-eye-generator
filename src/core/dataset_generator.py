import asyncio
import os
import csv
import random
from datetime import datetime

from src.config.settings import BASE_DIR
from src.consts.magic_eye_assets import MAGIC_EYE_ASSETS
from src.services.magic_eye_service import MagicEyeService


async def generate_magic_eye_dataset(num_images_per_asset: int = 300):
    """
    매직아이 이미지와 정답(Depth Map)을 생성하여
    학습용 데이터셋(CSV 메타데이터 포함) 구조로 저장합니다.
    """

    # 1. 경로 설정 (outputs 대신 데이터셋 성격을 강조하기 위해 datasets 폴더 사용)
    dataset_base_dir = os.path.join(BASE_DIR, "datasets")
    os.makedirs(dataset_base_dir, exist_ok=True)

    # 2. 메타데이터 CSV 파일 초기화 (기존 파일이 없으면 헤더 생성)
    metadata_path = os.path.join(dataset_base_dir, "metadata.csv")
    file_exists = os.path.exists(metadata_path)

    with open(metadata_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'asset_id',  # 사물 ID (라벨링용)
                'display_name',  # 사물 이름
                'description',  # LLM이 생성한 상세 묘사 (데이터 특징 기록)
                'problem_path',  # 입력 이미지 경로 (X)
                'answer_path',  # 정답 이미지 경로 (Y)
                'split',  # train / val / test 분할
                'created_at'  # 생성 일시
            ])

    # 3. 서비스 초기화
    print("🔄 모델 로딩 중... 모델이 메모리에 올라갈 때까지 잠시만 기다려주세요.")
    service = MagicEyeService()

    total_assets = len(MAGIC_EYE_ASSETS)
    total_expected = total_assets * num_images_per_asset
    completed_count = 0

    print(f"🚀 총 {total_assets}개의 에셋에 대해 각각 {num_images_per_asset}장씩 생성을 시작합니다.")

    # 4. 에셋별 반복 생성
    for a_idx, asset in enumerate(MAGIC_EYE_ASSETS, 1):
        asset_id = asset['id']
        asset_display_name = asset['display_name']

        # 에셋별 폴더 생성
        asset_output_dir = os.path.join(dataset_base_dir, asset_id)
        os.makedirs(asset_output_dir, exist_ok=True)

        print(f"\n📦 [{a_idx}/{total_assets}] 카테고리: {asset_display_name} ({asset_id})")

        for i in range(num_images_per_asset):
            # 파일명 규칙: 중복 생성 방지를 위해 순번 사용
            prob_filename = f"{asset_id}_{i + 1}_prob.png"
            ans_filename = f"{asset_id}_{i + 1}_ans.png"
            prob_path = os.path.join(asset_output_dir, prob_filename)
            ans_path = os.path.join(asset_output_dir, ans_filename)

            # 중복 체크 (중단 후 재시작 가능)
            if os.path.exists(prob_path) and os.path.exists(ans_path):
                print(f"  [Image {i + 1}/{num_images_per_asset}] 이미 존재함. 건너뜁니다.")
                completed_count += 1
                continue

            start_time = datetime.now()
            print(f"  [Image {i + 1}/{num_images_per_asset}] 생성 중...")

            try:
                # 5. LLM을 통한 다양한 프롬프트 확보 (데이터 다양성 증대)
                llm_prompts = await service.prompt_agent.get_diverse_prompts(asset_display_name, count=1)
                current_prompt = llm_prompts[0] if llm_prompts else asset['prompt']

                # 6. 이미지 생성 실행
                generated_data = await service.generate_specific_game(asset, current_prompt)

                # 7. 파일 저장
                with open(prob_path, "wb") as f:
                    f.write(generated_data.problem_image)
                with open(ans_path, "wb") as f:
                    f.write(generated_data.answer_image)

                # 8. 데이터셋 분할 (80% Train, 10% Val, 10% Test)
                rand_val = random.random()
                if rand_val < 0.8:
                    split = 'train'
                elif rand_val < 0.9:
                    split = 'val'
                else:
                    split = 'test'

                # 9. 메타데이터 CSV 기록
                with open(metadata_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        asset_id,
                        asset_display_name,
                        current_prompt,
                        os.path.relpath(prob_path, dataset_base_dir),  # 상대 경로로 저장 (이식성)
                        os.path.relpath(ans_path, dataset_base_dir),
                        split,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ])

                duration = (datetime.now() - start_time).seconds
                completed_count += 1
                progress = (completed_count / total_expected) * 100
                print(f"    ✅ 완료! ({duration}s) | 분할: {split.upper()} | 진행률: {progress:.1f}%")

            except Exception as e:
                print(f"    ❌ {asset_id} 생성 중 오류 발생: {str(e)}")
                await asyncio.sleep(2)  # 오류 발생 시 잠시 휴식

    print("\n✨ 데이터셋 구축 완료! 모든 이미지가 'datasets' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    # 배치 생성 실행 (각 에셋당 300장씩)
    asyncio.run(generate_magic_eye_dataset(num_images_per_asset=300))