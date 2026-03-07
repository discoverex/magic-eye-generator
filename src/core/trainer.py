import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from typing import List, Dict

from src.config.settings import BASE_DIR
from src.consts.magic_eye_assets import MAGIC_EYE_ASSETS


class MagicEyeDataset(Dataset):
    """
    CSV 메타데이터를 기반으로 매직아이 이미지를 로드하는 데이터셋 클래스
    """
    def __init__(self, csv_file: str, root_dir: str, transform=None, cat_to_idx: Dict[str, int] = None):
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.cat_to_idx = cat_to_idx

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.root_dir, "datasets", row['problem_path'])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"❌ 이미지 로드 실패: {img_path} | 에러: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        label = self.cat_to_idx[row['asset_id']]

        if self.transform:
            image = self.transform(image)

        return image, label


class MagicEyeTrainer:
    """
    AI 플레이어의 단계별 학습을 담당하는 클래스
    """
    def __init__(self):
        self.model_save_dir = BASE_DIR / "models" / "players"
        self.categories = [asset["id"] for asset in MAGIC_EYE_ASSETS]
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 기본 전처리 설정
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        os.makedirs(self.model_save_dir, exist_ok=True)

    def _prepare_dataloader(self, data_ratio: float, batch_size: int) -> DataLoader:
        """
        데이터 비율에 맞게 샘플링된 DataLoader를 준비합니다.
        """
        full_dataset = MagicEyeDataset(
            csv_file=os.path.join(BASE_DIR, "datasets/metadata.csv"),
            root_dir=str(BASE_DIR),
            transform=self.transform,
            cat_to_idx=self.cat_to_idx
        )

        num_samples = max(int(len(full_dataset) * data_ratio), 1)
        indices = torch.randperm(len(full_dataset))[:num_samples]
        train_subset = Subset(full_dataset, indices)

        return DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    def _init_model(self) -> nn.Module:
        """
        ResNet-18 모델을 초기화합니다.
        """
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(self.categories))
        return model.to(self.device)

    def train_level(self, level: int, data_ratio: float, batch_size: int = 64, epochs: int = 30):
        """
        특정 레벨의 AI를 훈련시킵니다.
        """
        print(f"\n{'=' * 40}")
        print(f"🚀 AI Player Level {level} 훈련 시작 (장치: {self.device})")
        print(f"📊 데이터 사용 비율: {data_ratio * 100}% | Batch: {batch_size} | Epochs: {epochs}")
        print(f"{'=' * 40}")

        train_loader = self._prepare_dataloader(data_ratio, batch_size)
        model = self._init_model()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")

        # 모델 저장
        save_path = self.model_save_dir / f"ai_lv{level}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"✅ Level {level} 훈련 완료 및 저장: {save_path}")

    def run_full_training(self):
        """
        1부터 10레벨까지 전체 시나리오를 훈련합니다.
        GPU 사용 가능 여부에 따라 훈련 파라미터를 자동으로 조정합니다.
        """
        is_cuda = torch.cuda.is_available()

        params = {
            'batch_size': 64 if is_cuda else 16,
            'epochs': 30 if is_cuda else 5
        }

        device_name = "GPU (CUDA)" if is_cuda else "CPU"
        print(f"🚀 {device_name} 환경에서 훈련을 시작합니다.")

        for i in range(1, 11):
            level = i
            ratio = i / 10
            try:
                self.train_level(level, ratio, **params)
            except Exception as e:
                print(f"❌ Level {level} 훈련 중 오류 발생: {e}")


if __name__ == "__main__":
    trainer = MagicEyeTrainer()
    # 전체 훈련 실행
    trainer.run_full_training()
