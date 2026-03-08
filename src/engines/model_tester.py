import os
from typing import Dict

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
    훈련된 AI 플레이어 모델의 최종 성능을 테스트(Test)하고 시각화하는 클래스
    """
    def __init__(self):
        self.model_dir = BASE_DIR / "models" / "players"
        self.result_dir = BASE_DIR / "test_results"
        self.categories = [asset["id"] for asset in MAGIC_EYE_ASSETS]
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        self.idx_to_display = {i: asset["display_name"] for i, asset in enumerate(MAGIC_EYE_ASSETS)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 테스트용 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        os.makedirs(self.result_dir, exist_ok=True)

    def _prepare_test_loader(self, batch_size: int = 32) -> DataLoader:
        """
        최종 테스트용 데이터셋(split='test')을 로드합니다.
        """
        test_dataset = MagicEyeDataset(
            csv_file=os.path.join(BASE_DIR, "datasets/metadata.csv"),
            root_dir=str(BASE_DIR),
            split='test',
            transform=self.transform,
            cat_to_idx=self.cat_to_idx
        )
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def _load_model(self, level: int) -> nn.Module:
        model_path = self.model_dir / f"ai_lv{level}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")

        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, len(self.categories))
        
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model.to(self.device)

    def test_level(self, level: int) -> float:
        """
        특정 레벨 모델의 최종 테스트 정확도를 평가하고 반환합니다.
        """
        print(f"\n{'-' * 45}")
        print(f"🧪 AI Player Level {level} 최종 테스트 중...")

        try:
            model = self._load_model(level)
            test_loader = self._prepare_test_loader()
        except Exception as e:
            print(f"⚠️ 테스트 준비 중 오류 발생: {e}")
            return 0.0

        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            # 테스트 진행바 설정 (leave=False로 테스트 완료 후 바 삭제)
            test_pbar = tqdm(test_loader, desc=f"레벨 {level} 성능 평가", unit="배치", leave=False)
            for images, labels in test_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0
        tqdm.write(f"✅ Level {level} Test Accuracy: {accuracy:.2f}%")
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

        results = {}
        for lv in levels:
            accuracy = self.test_level(lv)
            results[lv] = accuracy

        self.visualize_test_results(results)


if __name__ == "__main__":
    tester = ModelTester()
    tester.run_full_test()
