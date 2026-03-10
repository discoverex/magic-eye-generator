import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def get_dataset_stats(metadata_path: str):
    """전체 split 통계를 계산하여 반환합니다."""
    if not os.path.exists(metadata_path):
        return None, None

    split_counts = defaultdict(int)
    total_count = 0

    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_counts[row['split']] += 1
            total_count += 1

    return split_counts, total_count

def get_asset_stats(metadata_path: str):
    """asset_id별 split 통계를 계산하여 반환합니다."""
    if not os.path.exists(metadata_path):
        return None

    # asset_stats[asset_id][split] = count
    asset_stats = defaultdict(lambda: defaultdict(int))
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            asset_stats[row['asset_id']][row['split']] += 1
            
    return asset_stats

def print_and_visualize_stats(metadata_path: str, title="전체 데이터셋 분배 현황"):
    """전체 통계를 출력하고 시각화합니다."""
    split_counts, total_count = get_dataset_stats(metadata_path)

    if split_counts is None:
        print(f"❌ 파일을 찾을 수 없습니다: {metadata_path}")
        return

    train_count = split_counts.get('train', 0)
    val_count = split_counts.get('val', 0)
    test_count = split_counts.get('test', 0)

    train_pct = (train_count / total_count * 100) if total_count > 0 else 0
    val_pct = (val_count / total_count * 100) if total_count > 0 else 0
    test_pct = (test_count / total_count * 100) if total_count > 0 else 0

    print("-" * 45)
    print(f"📊 {title} (총 {total_count}개):")
    print(f"  - 🟢 Train: {train_count:>5}개 ({train_pct:>5.1f}%)")
    print(f"  - 🟡 Val  : {val_count:>5}개 ({val_pct:>5.1f}%)")
    print(f"  - 🔴 Test : {test_count:>5}개 ({test_pct:>5.1f}%)")
    print("-" * 45)

    labels = ['Train', 'Val', 'Test']
    sizes = [train_count, val_count, test_count]
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']

    valid_labels = [l for l, s in zip(labels, sizes) if s > 0]
    valid_sizes = [s for s in sizes if s > 0]
    valid_colors = [c for c, s in zip(colors, sizes) if s > 0]

    plt.figure(figsize=(8, 6))
    plt.pie(valid_sizes, labels=valid_labels, autopct='%1.1f%%', startangle=140, colors=valid_colors, shadow=True)
    plt.title(title)
    plt.axis('equal')
    plt.show()

def print_and_visualize_asset_stats(metadata_path: str, title="에셋별 데이터셋 분배 현황"):
    """에셋별 통계를 출력하고 시각화합니다."""
    asset_stats = get_asset_stats(metadata_path)
    if not asset_stats:
        print(f"❌ 데이터를 불러올 수 없습니다: {metadata_path}")
        return

    asset_ids = sorted(asset_stats.keys())
    train_counts = []
    val_counts = []
    test_counts = []

    print("-" * 60)
    print(f"📊 {title}:")
    print(f"{'Asset ID':<15} | {'Train':^12} | {'Val':^12} | {'Test':^12}")
    print("-" * 60)

    for aid in asset_ids:
        t = asset_stats[aid].get('train', 0)
        v = asset_stats[aid].get('val', 0)
        te = asset_stats[aid].get('test', 0)
        total = t + v + te
        
        train_counts.append(t)
        val_counts.append(v)
        test_counts.append(te)
        
        t_p = (t/total*100) if total > 0 else 0
        v_p = (v/total*100) if total > 0 else 0
        te_p = (te/total*100) if total > 0 else 0
        
        print(f"{aid:<15} | {t:>4} ({t_p:>4.1f}%) | {v:>4} ({v_p:>4.1f}%) | {te:>4} ({te_p:>4.1f}%)")
    
    print("-" * 60)

    # 시각화 (Stacked Bar Chart)
    x = np.arange(len(asset_ids))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x, train_counts, width, label='Train', color='#2ecc71')
    ax.bar(x, val_counts, width, bottom=train_counts, label='Val', color='#f1c40f')
    ax.bar(x, test_counts, width, bottom=np.array(train_counts)+np.array(val_counts), label='Test', color='#e74c3c')

    ax.set_ylabel('데이터 개수')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(asset_ids, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from src.config.settings import BASE_DIR
    target_path = os.path.join(BASE_DIR, "datasets", "metadata.csv")
    print_and_visualize_stats(target_path)
    print_and_visualize_asset_stats(target_path)
