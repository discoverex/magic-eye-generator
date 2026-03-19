import os
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.path_utils import ensure_local_path


class MagicEyeDataset(Dataset):
    """
    metadata.csv 데이터를 기반으로 매직아이 이미지와 메타데이터를 로드하는 PyTorch Dataset 클래스
    """
    def __init__(
        self, 
        csv_file: str, 
        root_dir: str, 
        split: Optional[str] = None, 
        transform=None, 
        cat_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            csv_file: metadata.csv 파일 경로
            root_dir: 프로젝트 루트 경로 (datasets 폴더가 포함된 곳)
            split: 'train', 'val', 'test' 중 하나를 선택하여 필터링 (None이면 전체)
            transform: 이미지 전처리 함수
            cat_to_idx: 카테고리 ID를 인덱스로 변환하는 딕셔너리 (분류 학습용)
        """
        # 1. 메타데이터 로드
        df = pd.read_csv(csv_file)
        
        # 2. split 필터링 (train, val, test)
        if split:
            self.metadata = df[df['split'] == split].reset_index(drop=True)
        else:
            self.metadata = df
            
        self.root_dir = root_dir
        self.transform = transform
        self.cat_to_idx = cat_to_idx

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        학습 시에는 이미지와 해당 이미지의 클래스 인덱스를 반환합니다.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata.iloc[idx]
        
        # 1. 이미지 로드 (문제 이미지)
        # metadata.csv의 problem_path는 datasets/ 기준 상대경로임
        # OS 호환성을 위해 경로 정규화 수행
        normalized_rel_path = ensure_local_path(row['problem_path'])
        img_path = os.path.join(self.root_dir, "datasets", normalized_rel_path)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"❌ 이미지 로드 실패: {img_path} | 에러: {e}")
            # 에러 발생 시 검은색 기본 이미지 반환
            image = Image.new('RGB', (224, 224), color='black')

        # 2. 전처리 적용
        if self.transform:
            image = self.transform(image)

        # 3. 라벨링 (cat_to_idx가 있으면 인덱스 반환, 없으면 asset_id 반환)
        if self.cat_to_idx:
            label = self.cat_to_idx[row['asset_id']]
        else:
            label = row['asset_id']

        return image, label

    def get_row_details(self, idx: int) -> Dict:
        """
        특정 인덱스의 상세 메타데이터(모든 컬럼)를 딕셔너리로 반환합니다.
        """
        return self.metadata.iloc[idx].to_dict()
