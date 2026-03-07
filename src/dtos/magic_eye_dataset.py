import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MagicEyeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 1. 이미지 로드 (문제 이미지)
        img_path = os.path.join(self.root_dir, self.metadata.iloc[idx]['problem_path'])
        image = Image.open(img_path).convert("RGB")

        # 2. 정답 라벨 (사물 이름)
        label_name = self.metadata.iloc[idx]['display_name']
        # TODO: label_name을 숫자로 바꾸는 매핑 로직 필요

        # 3. 상세 설명 (텍스트 데이터)
        description = self.metadata.iloc[idx]['description']

        if self.transform:
            image = self.transform(image)

        return image, label_name, description