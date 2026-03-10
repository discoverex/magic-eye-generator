import os
import multiprocessing
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from src.config.settings import BASE_DIR
from src.consts.magic_eye_assets import MAGIC_EYE_ASSETS
from src.dtos.magic_eye_dataset import MagicEyeDataset


class ModelTester:
    """
    훈련된 AI 플레이어 모델의 최종 성능을 테스트(Test)하고 시각화하는 클래스.
    GPU 병렬 처리 및 추론 최적화가 적용되었습니다.
    """
    def __init__(self):
        self.model_dir = BASE_DIR / "models" / "players"
        self.result_dir = BASE_DIR / "test_results"
        self.categories = [asset["id"] for asset in MAGIC_EYE_ASSETS]
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        self.idx_to_display = {i: asset["display_name"] for i, asset in enumerate(MAGIC_EYE_ASSETS)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # CPU 코어 수 기반 병렬 워커 수 설정
        self.num_workers = min(multiprocessing.cpu_count(), 8)

        # 테스트용 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        os.makedirs(self.result_dir, exist_ok=True)

    def _prepare_test_loader(self, batch_size: int = 128) -> DataLoader:
        """
        최종 테스트용 데이터셋(split='test')을 로드합니다.
        GPU 전송 최적화 및 병렬 로딩이 적용됩니다.
        """
        metadata_path = os.path.join(BASE_DIR, "datasets", "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"❌ 메타데이터 파일을 찾을 수 없습니다: {metadata_path}")

        test_dataset = MagicEyeDataset(
            csv_file=metadata_path,
            root_dir=str(BASE_DIR),
            split='test',
            transform=self.transform,
            cat_to_idx=self.cat_to_idx
        )
        
        return DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def _load_model(self, level: int) -> nn.Module:
        """모델 파일을 로드하고 평가 모드로 전환합니다."""
        model_path = self.model_dir / f"ai_lv{level}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")

        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, len(self.categories))
        
        # 가중치 로드
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        # PyTorch 2.0+ 컴파일 최적화 (추론 속도 향상)
        if hasattr(torch, 'compile') and os.name != 'nt':
            try:
                model = torch.compile(model)
            except Exception: pass
            
        return model

    def test_level(self, level: int, test_loader: Optional[DataLoader] = None) -> float:
        """
        특정 레벨 모델의 최종 테스트 정확도를 평가합니다. AMP 추론이 적용되었습니다.
        """
        print(f"\n{'-' * 50}")
        print(f"🧪 AI Player Level {level} 최종 테스트 시작 (장치: {self.device})")

        try:
            model = self._load_model(level)
            if test_loader is None:
                test_loader = self._prepare_test_loader()
        except Exception as e:
            print(f"⚠️ 테스트 중 오류 발생: {e}")
            return 0.0

        correct = 0
        total = 0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"  └ 레벨 {level} 성능 평가", unit="배치", leave=False)
            for images, labels in test_pbar:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                # AMP Autocast 적용 (추론 가속)
                with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
                    outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0
        tqdm.write(f"✅ Level {level} Test Accuracy: {accuracy:.2f}% (Total: {total} images)")
        return accuracy

    def visualize_test_results(self, results: Dict[int, float]):
        """
        레벨별 테스트 정확도 결과를 그래프로 시각화하여 저장합니다.
        """
        if not results:
            print("📉 시각화할 데이터가 없습니다.")
            return

        levels = sorted(results.keys())
        accuracies = [results[lv] for lv in levels]

        plt.figure(figsize=(12, 7))
        plt.bar(levels, accuracies, color='skyblue', alpha=0.8)
        plt.plot(levels, accuracies, marker='D', linestyle='--', color='red', linewidth=1)
        
        plt.title('Final AI Player Performance (Test Set)', fontsize=16, pad=20)
        plt.xlabel('AI Level', fontsize=12)
        plt.ylabel('Test Accuracy (%)', fontsize=12)
        plt.xticks(levels)
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)
        plt.ylim(0, 105)

        for i, acc in enumerate(accuracies):
            plt.text(levels[i], acc + 1, f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')

        save_path = self.result_dir / "test_accuracy_summary.png"
        plt.savefig(save_path)
        plt.close()
        
        print(f"\n📊 테스트 시각화 완료! 결과 저장됨: {save_path}")
        
        try:
            if os.name == 'nt':
                os.startfile(save_path)
            else:
                import subprocess
                opener = "open" if os.uname().sysname == "Darwin" else "xdg-open"
                subprocess.call([opener, str(save_path)])
        except Exception:
            print("💡 이미지를 자동으로 열 수 없습니다. 직접 확인해 주세요.")

    def run_full_test(self):
        """
        모든 모델 레벨에 대해 최종 테스트를 수행하고 시각화합니다.
        """
        levels = []
        if self.model_dir.exists():
            for f in os.listdir(self.model_dir):
                if f.startswith("ai_lv") and f.endswith(".pth"):
                    try:
                        lv = int(f.replace("ai_lv", "").replace(".pth", ""))
                        levels.append(lv)
                    except ValueError: continue
        
        levels.sort()
        
        if not levels:
            print("📁 테스트할 모델 파일이 'models/players/'에 없습니다.")
            return

        # 테스트 로더를 한 번만 준비하여 재사용 (속도 향상)
        try:
            test_loader = self._prepare_test_loader(batch_size=128)
        except Exception as e:
            print(f"❌ 테스트 준비 실패: {e}")
            return

        results = {}
        for lv in levels:
            accuracy = self.test_level(lv, test_loader=test_loader)
            results[lv] = accuracy

        self.visualize_test_results(results)


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    
    tester = ModelTester()
    tester.run_full_test()
