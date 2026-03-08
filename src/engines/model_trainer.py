import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from tqdm import tqdm

from src.config.settings import BASE_DIR
from src.consts.magic_eye_assets import MAGIC_EYE_ASSETS
from src.dtos.magic_eye_dataset import MagicEyeDataset


class MagicEyeTrainer:
    """
    AI 플레이어의 단계별 학습을 담당하는 클래스.
    이전 레벨에서 학습된 가중치를 이어서 학습하는 Continual Learning 방식을 지원합니다.
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

    def _prepare_dataloaders(self, data_ratio: float, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """
        훈련용(train) 및 검증용(val) DataLoader를 준비합니다.
        """
        # 훈련 데이터셋 (비율 적용)
        train_dataset = MagicEyeDataset(
            csv_file=os.path.join(BASE_DIR, "datasets/metadata.csv"),
            root_dir=str(BASE_DIR),
            split='train',
            transform=self.transform,
            cat_to_idx=self.cat_to_idx
        )
        num_train_samples = max(int(len(train_dataset) * data_ratio), 1)
        train_indices = torch.randperm(len(train_dataset))[:num_train_samples]
        train_subset = Subset(train_dataset, train_indices)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        # 검증 데이터셋 (전체 사용)
        val_dataset = MagicEyeDataset(
            csv_file=os.path.join(BASE_DIR, "datasets/metadata.csv"),
            root_dir=str(BASE_DIR),
            split='val',
            transform=self.transform,
            cat_to_idx=self.cat_to_idx
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def _init_model(self, model_path: Optional[str] = None) -> nn.Module:
        """
        ResNet-18 모델을 초기화합니다. model_path가 제공되면 해당 가중치를 로드합니다.
        """
        if model_path and os.path.exists(model_path):
            print(f"🔄 이전 레벨 가중치 로드 중: {model_path}")
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(self.categories))
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("🆕 신규 모델 초기화 (ImageNet 사전 학습 가중치 사용)")
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, len(self.categories))
            
        return model.to(self.device)

    def train_level(self, level: int, data_ratio: float, batch_size: int = 64, epochs: int = 50, patience: int = 5):
        """
        특정 레벨의 AI를 훈련시킵니다. 이전 레벨의 가중치가 있다면 이를 상속받습니다.
        """
        print(f"\n{'=' * 50}")
        print(f"🚀 AI 플레이어 레벨 {level} 훈련 시작 (장치: {self.device})")
        print(f"📊 데이터 사용 비율: {data_ratio * 100}% | 조기 종료 대기(Patience): {patience}")
        print(f"{'=' * 50}")

        # 이전 레벨 모델 경로 확인
        previous_level_path = None
        if level > 1:
            prev_path = self.model_save_dir / f"ai_lv{level - 1}.pth"
            if prev_path.exists():
                previous_level_path = str(prev_path)

        train_loader, val_loader = self._prepare_dataloaders(data_ratio, batch_size)
        model = self._init_model(previous_level_path)
        
        criterion = nn.CrossEntropyLoss()
        # 가중치 상속 시에는 학습률을 낮추어 안정성을 확보합니다.
        lr = 0.0001 if level == 1 else 0.00005 
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        save_path = self.model_save_dir / f"ai_lv{level}.pth"

        for epoch in range(epochs):
            # --- 학습 단계 ---
            model.train()
            train_loss = 0.0
            
            # 학습 진행바 설정 (position=1, leave=False로 에포크마다 갱신)
            train_pbar = tqdm(train_loader, desc=f"  └ 에포크 {epoch+1} 학습", unit="배치", position=1, leave=False)
            for images, labels in train_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix(손실=f"{loss.item():.4f}")

            avg_train_loss = train_loss / len(train_loader)

            # --- 검증 단계 ---
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            # 검증 진행바 설정 (동일한 position=1 사용으로 학습 바 자리를 대체)
            val_pbar = tqdm(val_loader, desc=f"  └ 에포크 {epoch+1} 검증", unit="배치", position=1, leave=False)
            with torch.no_grad():
                for images, labels in val_pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    val_pbar.set_postfix(손실=f"{loss.item():.4f}")

            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total if total > 0 else 0

            # 결과 출력을 tqdm.write로 하여 진행바 위치를 깨뜨리지 않음
            tqdm.write(f"📊 [Epoch {epoch + 1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # --- 조기 종료 및 최적 모델 저장 ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), save_path)
                tqdm.write(f"  ✨ Best Model Saved! (Loss: {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    tqdm.write(f"  🛑 Early Stopping triggered after {patience} epochs of no improvement.")
                    break

        print(f"✅ 레벨 {level} 훈련 완료. 최적 모델 저장 위치: {save_path}")

    def run_full_training(self):
        """
        1부터 10레벨까지 전체 시나리오를 훈련합니다.
        """
        for i in range(1, 11):
            level = i
            ratio = i / 10
            try:
                self.train_level(level, ratio, batch_size=64, epochs=i*5, patience=5)
            except Exception as e:
                print(f"❌ 레벨 {level} 훈련 중 오류 발생: {e}")
                continue


if __name__ == "__main__":
    trainer = MagicEyeTrainer()
    trainer.run_full_training()
