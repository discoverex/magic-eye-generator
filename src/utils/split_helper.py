from typing import Literal

def get_split_name(index: int, total: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Literal['train', 'val', 'test']:
    """
    인덱스와 전체 개수를 기반으로 데이터셋 split(train, val, test)을 결정합니다.
    인덱스는 0-based 또는 1-based 모두 대응 가능하도록 처리합니다. (내부적으로 0-based로 변환)
    """
    # 1-based 인덱스가 들어올 경우를 대비해 0-based로 변환 (보통 1부터 시작하는 경우가 많음)
    # 하지만 0부터 시작할 수도 있으므로, 안전하게 처리
    normalized_idx = index if index < total else index - 1
    
    # 더 정확한 방법: index가 1부터 시작한다고 가정하고 0-based로 정규화
    # 만약 index가 1~300 이라면, normalized_idx는 0~299
    # dataset_generator.py는 1부터 시작하므로 (i-1)을 사용해야 함.
    
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    if normalized_idx < train_end:
        return 'train'
    elif normalized_idx < val_end:
        return 'val'
    else:
        return 'test'

def get_split_from_index(idx_1_based: int, total: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Literal['train', 'val', 'test']:
    """1부터 시작하는 인덱스를 기반으로 split을 반환합니다."""
    idx_0_based = idx_1_based - 1
    return get_split_name(idx_0_based, total, train_ratio, val_ratio)
