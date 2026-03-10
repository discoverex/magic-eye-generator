import os
import multiprocessing
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
    GPU 병렬 처리 및 파이프라인 최적화가 적용되었습니다.
    """
    def __init__(self):
        self.model_save_dir = BASE_DIR / "models" / "players"
        self.categories = [asset["id"] for asset in MAGIC_EYE_ASSETS]
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # CPU 코어 수 기반 병렬 워커 수 설정
        self.num_workers = min(multiprocessing.cpu_count(), 8)
        
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
        데이터 불균형 해결을 위해 WeightedRandomSampler를 적용합니다.
        """
        metadata_path = os.path.join(BASE_DIR, "datasets", "metadata.csv")
        
        # 1. 훈련 데이터셋 로드
        train_dataset = MagicEyeDataset(
            csv_file=metadata_path,
            root_dir=str(BASE_DIR),
            split='train',
            transform=self.transform,
            cat_to_idx=self.cat_to_idx
        )
        
        # 2. 클래스별 가중치 계산 (불균형 해소)
        # 각 샘플의 클래스(asset_id) 빈도수를 계산하여 역수를 가중치로 부여
        labels = [self.cat_to_idx[row['asset_id']] for _, row in train_dataset.metadata.iterrows()]
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1. / class_counts.float()
        sample_weights = class_weights[labels]

        # 3. 샘플러 설정
        # data_ratio에 따라 전체 학습 데이터 중 사용할 샘플 수 결정
        num_train_samples = max(int(len(train_dataset) * data_ratio), 1)
        
        # WeightedRandomSampler는 각 클래스를 균등하게 뽑도록 유도함
        # replacement=True로 설정하여 데이터가 적은 클래스는 중복 추출될 수 있게 함
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_train_samples,
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if self.num_workers > 0 else False
        )

        # 4. 검증 데이터셋 (검증은 원래 분포를 확인하기 위해 셔플 없이 로드)
        val_dataset = MagicEyeDataset(
            csv_file=metadata_path,
            root_dir=str(BASE_DIR),
            split='val',
            transform=self.transform,
            cat_to_idx=self.cat_to_idx
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if self.num_workers > 0 else False
        )

        return train_loader, val_loader

    def _init_model(self, model_path: Optional[str] = None) -> nn.Module:
        """
        ResNet-18 모델을 초기화합니다. model_path가 제공되면 해당 가중치를 로드합니다.
        PyTorch 2.0+ 의 경우 torch.compile()을 시도합니다.
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
            
        model = model.to(self.device)
        
        # PyTorch 2.0+ 컴파일 최적화 (Linux/Windows 지원)
        if hasattr(torch, 'compile') and os.name != 'nt': # Windows에서는 아직 일부 제약이 있을 수 있음
            try:
                print("⚡ PyTorch 2.0 Compile 적용 중...")
                model = torch.compile(model)
            except Exception as e:
                print(f"⚠️ 컴파일 실패 (기본 모드 실행): {e}")
                
        return model

    def train_level(self, level: int, data_ratio: float, batch_size: int = 64, epochs: int = 50, patience: int = 5):
        """
        특정 레벨의 AI를 훈련시킵니다. AMP(혼합 정밀도) 학습이 적용되었습니다.
        """
        print(f"\n{'=' * 60}")
        print(f"🚀 AI 플레이어 레벨 {level} 훈련 시작 (장치: {self.device})")
        print(f"📊 데이터 사용 비율: {data_ratio * 100:.1f}% | 워커 수: {self.num_workers}")
        print(f"{'=' * 60}")

        # 이전 레벨 모델 경로 확인
        previous_level_path = None
        if level > 1:
            prev_path = self.model_save_dir / f"ai_lv{level - 1}.pth"
            if prev_path.exists():
                previous_level_path = str(prev_path)

        train_loader, val_loader = self._prepare_dataloaders(data_ratio, batch_size)
        model = self._init_model(previous_level_path)
        
        criterion = nn.CrossEntropyLoss()
        lr = 0.0001 if level == 1 else 0.00005 
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 혼합 정밀도(AMP) 설정을 위한 스케일러
        scaler = torch.amp.GradScaler('cuda' if self.device.type == 'cuda' else 'cpu')

        best_val_loss = float('inf')
        epochs_no_improve = 0
        save_path = self.model_save_dir / f"ai_lv{level}.pth"

        for epoch in range(epochs):
            # --- 학습 단계 ---
            model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"  └ 에포크 {epoch+1} 학습", unit="배치", position=1, leave=False)
            for images, labels in train_pbar:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # AMP Autocast 적용
                with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                train_pbar.set_postfix(손실=f"{loss.item():.4f}")

            avg_train_loss = train_loss / len(train_loader)

            # --- 검증 단계 ---
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            val_pbar = tqdm(val_loader, desc=f"  └ 에포크 {epoch+1} 검증", unit="배치", position=1, leave=False)
            with torch.no_grad():
                for images, labels in val_pbar:
                    images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    val_pbar.set_postfix(손실=f"{loss.item():.4f}")

            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total if total > 0 else 0

            tqdm.write(f"📊 [Epoch {epoch + 1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # --- 조기 종료 및 최적 모델 저장 ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                
                # 모델 저장 (컴파일된 모델의 경우 _orig_mod 추출 필요할 수도 있음)
                save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                torch.save(save_model.state_dict(), save_path)
                tqdm.write(f"  ✨ Best Model Saved! (Loss: {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    tqdm.write(f"  🛑 Early Stopping triggered after {patience} epochs.")
                    break

        print(f"✅ 레벨 {level} 훈련 완료. 저장 위치: {save_path}")

    def run_full_training(self):
        """
        1부터 10레벨까지 전체 시나리오를 훈련합니다.
        """
        for i in range(1, 11):
            level = i
            ratio = min(i / 10, 1.0)
            try:
                # 레벨이 올라갈수록 배치 사이즈를 키워 GPU 효율을 극대화 (VRAM 허용 범위 내)
                batch_size = 64 if i < 5 else 128
                self.train_level(level, ratio, batch_size=batch_size, epochs=i*10, patience=7)
            except Exception as e:
                print(f"❌ 레벨 {level} 훈련 중 오류 발생: {e}")
                continue


if __name__ == "__main__":
    import multiprocessing
    # Windows 환경에서의 멀티프로세싱 지원을 위해 필요
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    trainer = MagicEyeTrainer()
    trainer.run_full_training()
