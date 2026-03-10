import csv
import random
import os
from collections import defaultdict

from src.config.settings import BASE_DIR
from src.utils.dataset_stats import print_and_visualize_stats, print_and_visualize_asset_stats
from src.utils.split_helper import get_split_name


def rebalance_dataset_split(metadata_path: str, train_ratio=0.8, val_ratio=0.1):
    if not os.path.exists(metadata_path):
        print(f"❌ 파일을 찾을 수 없습니다: {metadata_path}")
        return

    # 1. 기존 데이터 로드
    rows = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # 2. 에셋별 그룹화 (에셋별로 비율을 맞추기 위함)
    asset_groups = defaultdict(list)
    for row in rows:
        asset_groups[row['asset_id']].append(row)

    final_rows = []
    print(f"🔄 총 {len(rows)}개의 데이터를 재분배합니다...")

    # 3. 각 에셋 그룹별로 셔플 후 8:1:1 분배
    for asset_id, group in asset_groups.items():
        random.shuffle(group)  # 💡 순서 편향 제거

        total = len(group)
        for i, row in enumerate(group):
            row['split'] = get_split_name(i, total, train_ratio, val_ratio)
            final_rows.append(row)

    # 4. 결과 저장
    with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    # 5. 최종 통계 출력 및 시각화
    print(f"✅ 재분배 완료! (Asset 종류: {len(asset_groups)}종)")
    print_and_visualize_stats(metadata_path, title="리밸런싱 후 전체 데이터셋 분배 현황")
    print_and_visualize_asset_stats(metadata_path, title="리밸런싱 후 에셋별 데이터셋 분배 현황")
    print(f"💡 목표 비율: {train_ratio*10:g}:{val_ratio*10:g}:{(1-train_ratio-val_ratio)*10:g}")


if __name__ == "__main__":
    target_path = os.path.join(BASE_DIR, "datasets", "metadata.csv")
    rebalance_dataset_split(target_path)
