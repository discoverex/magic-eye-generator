import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms

from src.config.settings import BASE_DIR
from src.consts.magic_eye_assets import MAGIC_EYE_ASSETS

MODEL_SAVE_DIR = BASE_DIR / "models" / "players"
CATEGORIES = [asset["id"] for asset in MAGIC_EYE_ASSETS]
cat_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}


class MagicEyeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # idx가 tensor나 list 형태일 경우를 대비해 정수로 변환
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.root_dir, "datasets", row['problem_path'])

        # 이미지 로드 시 에러 방지 처리
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"❌ 이미지 로드 실패: {img_path} | 에러: {e}")
            # 대체 이미지(검은 화면 등)를 반환하거나 에러를 던짐
            image = Image.new('RGB', (224, 224), color='black')

        label = cat_to_idx[row['asset_id']]

        if self.transform:
            image = self.transform(image)

        return image, label


def train_one_level(level, data_ratio, batch_size=4, epochs=3):
    print(f"\n{'=' * 30}")
    print(f"🚀 AI Player Level {level} 훈련 시작")
    print(f"📊 데이터 사용 비율: {data_ratio * 100}%")
    print(f"{'=' * 30}")

    # 데이터 전처리 및 증강 (데이터가 적을수록 증강이 중요함)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 데이터셋 준비
    full_dataset = MagicEyeDataset(
        csv_file=os.path.join(BASE_DIR, "datasets/metadata.csv"),
        root_dir=BASE_DIR,
        transform=transform
    )

    # 데이터 비율에 따른 샘플링 (난이도 구현 핵심)
    num_samples = max(int(len(full_dataset) * data_ratio), 1)
    indices = torch.randperm(len(full_dataset))[:num_samples]
    train_subset = Subset(full_dataset, indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    # 모델 설정 (ResNet-18)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CATEGORIES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # CPU 환경에선 batch_size가 작으니 lr=0.001로 변경
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 훈련 루프
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f}")

    # 모델 저장
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = MODEL_SAVE_DIR / f"ai_lv{level}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Level {level} 훈련 완료 및 저장: {save_path}")


if __name__ == "__main__":
    # 포트폴리오용 시나리오: 10가지 레벨의 AI 훈련
    # 1부터 10까지 반복하며 딕셔너리 생성
    levels = [{"level": i, "ratio": i / 10} for i in range(1, 11)]

    for lv_cfg in levels:
        try:
            # train_one_level(lv_cfg["level"], lv_cfg["ratio"]) # CPU test
            train_one_level(lv_cfg["level"], lv_cfg["ratio"], batch_size=64, epochs=30) # GPU
        except Exception as e:
            print(f"❌ Level {lv_cfg['level']} 훈련 중 오류 발생: {e}")