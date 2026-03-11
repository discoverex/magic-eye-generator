import os
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from tqdm import tqdm

from src.config.settings import BASE_DIR
from src.consts.magic_eye_assets import MAGIC_EYE_ASSETS
from src.services.gcp_storage_service import GCPStorageService


class ONNXConverter:
    """
    훈련된 PyTorch (.pth) 모델들을 ONNX 형식으로 변환하는 엔진 클래스.
    변환된 모델은 웹 브라우저(ONNX Runtime Web)나 Node.js에서 실행 가능합니다.
    """
    def __init__(self):
        self.model_dir = BASE_DIR / "models" / "players"
        self.onnx_dir = BASE_DIR / "models" / "onnx"
        self.categories = [asset["id"] for asset in MAGIC_EYE_ASSETS]
        self.device = torch.device("cpu") # 변환은 CPU에서 안정적으로 수행
        
        os.makedirs(self.onnx_dir, exist_ok=True)

    def _load_pytorch_model(self, level: int):
        """
        특정 레벨의 PyTorch 모델을 로드합니다.
        """
        model_path = self.model_dir / f"ai_lv{level}.pth"
        if not model_path.exists():
            return None

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.categories))
        
        # 가중치 로드
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def convert_all(self):
        """
        모든 레벨(1~10)의 모델을 ONNX로 변환합니다.
        """
        print(f"\n{'=' * 50}")
        print(f"🔄 PyTorch -> ONNX 모델 변환 시작")
        print(f"📁 소스: {self.model_dir}")
        print(f"📁 대상: {self.onnx_dir}")
        print(f"{'=' * 50}")

        converted_files = []
        
        # 변환을 위한 더미 입력 (ResNet-18 표준 입력 크기: 1, 3, 224, 224)
        dummy_input = torch.randn(1, 3, 224, 224)

        for level in range(1, 11):
            model = self._load_pytorch_model(level)
            if model is None:
                continue

            onnx_path = self.onnx_dir / f"ai_lv{level}.onnx"
            
            print(f"📦 레벨 {level} 변환 중...")
            try:
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12, # ONNX Runtime Web 호환성 고려
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                converted_files.append(onnx_path)
            except Exception as e:
                print(f"❌ 레벨 {level} 변환 실패: {e}")

        print(f"\n✨ 총 {len(converted_files)}개의 모델이 ONNX로 변환되었습니다.")
        return converted_files

    def upload_to_gcs(self, files):
        """
        변환된 ONNX 파일들을 GCP Storage 버킷에 업로드합니다.
        """
        if not files:
            print("📤 업로드할 파일이 없습니다.")
            return

        print(f"\n☁️ GCP Storage 업로드 시작...")
        gcs_service = GCPStorageService()
        
        # 버킷 내 'models/onnx/' 경로에 저장
        for file_path in tqdm(files, desc="GCS 업로드 중"):
            blob_name = f"models/onnx/{file_path.name}"
            gcs_service.upload_file(str(file_path), blob_name)
            
        print(f"✅ 모든 ONNX 모델이 GCS에 업로드되었습니다.")


if __name__ == "__main__":
    converter = ONNXConverter()
    files = converter.convert_all()
    
    if files:
        confirm = input("\n📤 변환된 파일을 GCS에 업로드하시겠습니까? (y/n): ").lower()
        if confirm == 'y':
            converter.upload_to_gcs(files)
