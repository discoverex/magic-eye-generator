import json
import os
from datetime import datetime
from typing import Dict, Optional, List, Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from src.config.settings import BASE_DIR
from src.consts.magic_eye_assets import MAGIC_EYE_ASSETS
from src.dtos.magic_eye_dataset import MagicEyeDataset
from src.utils.spinner import Spinner


class ModelTester:
    """
    훈련된 AI 플레이어 모델의 최종 성능을 테스트(Test)하고 시각화하는 클래스.
    GPU 병렬 처리 및 추론 최적화가 적용되었습니다. (PyTorch 및 ONNX 지원)
    """
    def __init__(self, model_type: str = "pth"):
        self.model_type = model_type.lower()
        if self.model_type == "onnx":
            self.model_dir = BASE_DIR / "models" / "onnx"
            import onnxruntime as ort
            self.ort = ort
        else:
            self.model_dir = BASE_DIR / "models" / "players"
            
        self.base_result_dir = BASE_DIR / "test_results"
        self.timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        self.run_dir = self.base_result_dir / self.timestamp
        
        self.categories = [asset["id"] for asset in MAGIC_EYE_ASSETS]
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        self.idx_to_display = {i: asset["display_name"] for i, asset in enumerate(MAGIC_EYE_ASSETS)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # CPU 코어 수 기반 병렬 워커 수 설정
        import multiprocessing
        self.num_workers = min(multiprocessing.cpu_count(), 8)

        # 테스트용 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 기존 결과 마이그레이션
        self._migrate_old_results()

    def _migrate_old_results(self):
        """기존 test_results 루트에 있는 파일들을 생성 일시 기반 폴더로 이동시킵니다."""
        if not self.base_result_dir.exists():
            return

        # 마이그레이션 기준 파일 확인
        target_file = self.base_result_dir / "test_accuracy_summary.png"
        # pth/onnx 구분이 포함된 파일명도 체크
        if not target_file.exists():
            summary_files = list(self.base_result_dir.glob("test_accuracy_summary_*.png"))
            if summary_files:
                target_file = summary_files[0]

        if target_file.exists():
            # 파일의 수정 시간을 기준으로 폴더 이름 생성 (생성 시간은 OS마다 다를 수 있어 mtime 사용)
            mtime = os.path.getmtime(target_file)
            dt = datetime.fromtimestamp(mtime)
            migration_folder_name = dt.strftime("%y%m%d_%H%M%S")
            migration_dir = self.base_result_dir / migration_folder_name
            
            # 마이그레이션 대상 파일들 (루트 직하의 파일들만)
            files_to_move = [f for f in os.listdir(self.base_result_dir) 
                             if os.path.isfile(self.base_result_dir / f)]
            
            if files_to_move:
                os.makedirs(migration_dir, exist_ok=True)
                print(f"📦 기존 테스트 결과를 마이그레이션합니다: -> {migration_folder_name}")
                import shutil
                for f_name in files_to_move:
                    shutil.move(str(self.base_result_dir / f_name), str(migration_dir / f_name))

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
            pin_memory=True if self.device.type == 'cuda' and self.model_type == 'pth' else False,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def _load_pth_model(self, level: int) -> nn.Module:
        """PyTorch 모델 파일을 로드하고 평가 모드로 전환합니다."""
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

    def _load_onnx_session(self, level: int):
        """ONNX 모델 파일을 로드하고 Inference Session을 생성합니다."""
        model_path = self.model_dir / f"ai_lv{level}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"❌ ONNX 모델 파일을 찾을 수 없습니다: {model_path}")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
        session = self.ort.InferenceSession(str(model_path), providers=providers)
        return session

    def test_level(self, level: int, test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        특정 레벨 모델의 최종 테스트 정확도를 평가합니다.
        """
        print(f"\n{'-' * 50}")
        print(f"🧪 AI Player Level {level} ({self.model_type.upper()}) 최종 테스트 시작 (장치: {self.device})")

        # 모델 파일 정보 수집
        ext = f".{self.model_type}"
        model_path = self.model_dir / f"ai_lv{level}{ext}"
        model_info = {"path": str(model_path), "exists": False}
        
        if model_path.exists():
            stat = os.stat(model_path)
            model_info.update({
                "exists": True,
                "filename": model_path.name,
                "size_bytes": stat.st_size,
                "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })

        try:
            if self.model_type == "pth":
                model = self._load_pth_model(level)
            else:
                session = self._load_onnx_session(level)
                input_name = session.get_inputs()[0].name

            if test_loader is None:
                test_loader = self._prepare_test_loader()
        except Exception as e:
            print(f"⚠️ 테스트 중 오류 발생: {e}")
            return {
                "level": level, 
                "accuracy": 0.0, 
                "total_images": 0, 
                "correct_images": 0, 
                "model_info": model_info,
                "per_asset_metrics": {}
            }

        correct = 0
        total = 0
        
        # 에셋별 정답 현황 기록용
        asset_correct = {cat: 0 for cat in self.categories}
        asset_total = {cat: 0 for cat in self.categories}
        
        if self.model_type == "pth":
            with torch.no_grad():
                test_pbar = tqdm(test_loader, desc=f"  └ 레벨 {level} pth 평가", unit="배치", leave=False)
                for images, labels in test_pbar:
                    images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
                        outputs = model(images)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # 배치 내 결과 집계
                    batch_correct = (predicted == labels)
                    correct += batch_correct.sum().item()
                    total += labels.size(0)
                    
                    # 에셋별 집계
                    for i in range(labels.size(0)):
                        label_idx = labels[i].item()
                        asset_id = self.categories[label_idx]
                        asset_total[asset_id] += 1
                        if batch_correct[i]:
                            asset_correct[asset_id] += 1
        else:
            # ONNX Inference
            test_pbar = tqdm(test_loader, desc=f"  └ 레벨 {level} onnx 평가", unit="배치", leave=False)
            for images, labels in test_pbar:
                # ONNX Runtime은 numpy array를 입력으로 받음
                images_np = images.numpy()
                outputs = session.run(None, {input_name: images_np})
                
                # outputs[0] is the logits
                import numpy as np
                predicted = np.argmax(outputs[0], axis=1)
                labels_np = labels.numpy()
                
                # 배치 내 결과 집계
                batch_correct = (predicted == labels_np)
                correct += batch_correct.sum()
                total += len(labels_np)
                
                # 에셋별 집계
                for i in range(len(labels_np)):
                    label_idx = labels_np[i]
                    asset_id = self.categories[label_idx]
                    asset_total[asset_id] += 1
                    if batch_correct[i]:
                        asset_correct[asset_id] += 1

        accuracy = 100 * correct / total if total > 0 else 0
        
        # 에셋별 상세 리포트 생성
        asset_metrics = {}
        for aid in self.categories:
            a_total = asset_total[aid]
            a_correct = asset_correct[aid]
            a_acc = (a_correct / a_total * 100) if a_total > 0 else 0
            asset_metrics[aid] = {
                "display_name": next((a['display_name'] for a in MAGIC_EYE_ASSETS if a['id'] == aid), aid),
                "total": a_total,
                "correct": a_correct,
                "accuracy": round(a_acc, 2)
            }

        tqdm.write(f"✅ Level {level} {self.model_type.upper()} Accuracy: {accuracy:.2f}% (Total: {total} images)")
        
        return {
            "level": level,
            "accuracy": round(accuracy, 2),
            "total_images": total,
            "correct_images": int(correct),
            "model_info": model_info,
            "per_asset_metrics": asset_metrics
        }

    def save_test_results_json(self, results: List[Dict[str, Any]]):
        """
        테스트 로그 전체를 JSON 파일로 저장합니다.
        """
        with Spinner("테스트 로그 JSON 저장 중..."):
            log_data = {
                "test_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": self.timestamp,
                "device": str(self.device),
                "model_type": self.model_type,
                "total_assets": len(self.categories),
                "levels_tested": len(results),
                "results": results
            }
            
            # 실제 파일 저장 직전에 폴더 생성 (빈 폴더 생성 방지)
            os.makedirs(self.run_dir, exist_ok=True)
            
            filename = f"test_logs_{self.model_type}_{self.timestamp}.json"
            save_path = self.run_dir / filename
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=4, ensure_ascii=False)
            
        print(f"📄 테스트 로그 저장 완료: {save_path}")

    def save_test_report_md(self, results: List[Dict[str, Any]]):
        """
        TestReportGenerator를 사용하여 상세 분석 보고서를 생성합니다.
        """
        from src.utils.test_report_generator import TestReportGenerator
        
        with Spinner("AI 분석 리포트 생성 중 (LLM 분석 포함)..."):
            log_data = {
                "test_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": self.timestamp,
                "device": str(self.device),
                "model_type": self.model_type,
                "results": results
            }
            
            generator = TestReportGenerator(run_dir=self.run_dir)
            generator.generate_from_data(log_data, self.timestamp)

    def visualize_test_results(self, results: List[Dict[str, Any]]):
        """
        레벨별 테스트 정확도 결과를 그래프로 시각화하여 저장합니다.
        """
        if not results:
            print("📉 시각화할 데이터가 없습니다.")
            return

        # 폴더 생성 확인
        os.makedirs(self.run_dir, exist_ok=True)
        
        levels = [r['level'] for r in results]
        accuracies = [r['accuracy'] for r in results]

        plt.figure(figsize=(12, 7))
        plt.bar(levels, accuracies, color='skyblue' if self.model_type == 'pth' else 'lightgreen', alpha=0.8)
        plt.plot(levels, accuracies, marker='D', linestyle='--', color='red', linewidth=1)
        
        plt.title(f'Final AI Player Performance ({self.model_type.upper()} Test Set)', fontsize=16, pad=20)
        plt.xlabel('AI Level', fontsize=12)
        plt.ylabel('Test Accuracy (%)', fontsize=12)
        plt.xticks(levels)
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)
        plt.ylim(0, 105)

        for i, acc in enumerate(accuracies):
            plt.text(levels[i], acc + 1, f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')

        filename = f"test_accuracy_summary_{self.model_type}_{self.timestamp}.png"
        save_path = self.run_dir / filename
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
        모든 모델 레벨에 대해 최종 테스트를 수행하고 시각화 및 로그 저장을 수행합니다.
        """
        levels = []
        ext = f".{self.model_type}"
        if self.model_dir.exists():
            for f in os.listdir(self.model_dir):
                if f.startswith("ai_lv") and f.endswith(ext):
                    try:
                        lv = int(f.replace("ai_lv", "").replace(ext, ""))
                        levels.append(lv)
                    except ValueError: continue
        
        levels.sort()
        
        if not levels:
            print(f"📁 테스트할 {self.model_type.upper()} 모델 파일이 '{self.model_dir}'에 없습니다.")
            return

        try:
            test_loader = self._prepare_test_loader(batch_size=128)
        except Exception as e:
            print(f"❌ 테스트 준비 실패: {e}")
            return

        all_results = []
        for lv in levels:
            result = self.test_level(lv, test_loader=test_loader)
            all_results.append(result)

        # JSON 로그 저장
        self.save_test_results_json(all_results)
        
        # Markdown 보고서 생성
        self.save_test_report_md(all_results)
        
        # 시각화
        self.visualize_test_results(all_results)


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass

    print("\n" + "="*50)
    print(" 🧪 AI 모델 성능 테스트 도구")
    print("="*50)
    print(" 1. PyTorch 모델 테스트 (.pth)")
    print(" 2. ONNX 모델 테스트 (.onnx)")
    print("-" * 50)
    choice = input("👉 테스트할 모델 유형을 선택하세요 (1 또는 2): ").strip()
    
    m_type = "pth" if choice == "1" else "onnx"
    if choice not in ["1", "2"]:
        print("⚠️ 기본값인 'pth'로 진행합니다.")
        m_type = "pth"

    tester = ModelTester(model_type=m_type)
    tester.run_full_test()
