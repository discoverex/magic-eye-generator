import os

import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

from src.config.settings import BASE_DIR, MODEL_BUCKET_NAME
from src.consts.magic_eye_assets import MAGIC_EYE_ASSETS
from src.services.gcp_storage_service import GCPStorageService


class ONNXConverter:
    """
    훈련된 PyTorch (.pth) 모델들을 ONNX 형식으로 변환하는 엔진 클래스.
    변환된 모델은 웹 브라우저(ONNX Runtime Web)나 Node.js에서 실행 가능합니다.
    """
    def __init__(self, bucket_name: str = MODEL_BUCKET_NAME):
        self.model_dir = BASE_DIR / "models" / "players"
        self.onnx_dir = BASE_DIR / "models" / "onnx"
        self.bucket_name = bucket_name
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

    def convert_all(self, quantize: bool = True):
        """
        모든 레벨(1~10)의 모델을 ONNX로 변환하며, 선택적으로 양자화를 수행합니다.
        """
        from src.utils.onnx_quantizer import ONNXQuantizer
        quantizer_tool = ONNXQuantizer()

        print(f"\n{'=' * 50}")
        print(f"🔄 PyTorch -> ONNX 모델 변환 시작 {'(양자화 포함)' if quantize else ''}")
        print(f"📁 소스: {self.model_dir}")
        print(f"📁 대상: {self.onnx_dir}")
        print(f"{'=' * 50}")

        converted_files = []
        dummy_input = torch.randn(1, 3, 224, 224)

        for level in range(1, 11):
            model = self._load_pytorch_model(level)
            if model is None:
                continue

            onnx_path = self.onnx_dir / f"ai_lv{level}.onnx"
            temp_onnx_path = self.onnx_dir / f"ai_lv{level}_fp32.onnx"
            
            print(f"📦 레벨 {level} 변환 중...")
            try:
                # 1. FP32 ONNX 모델로 내보내기
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(temp_onnx_path if quantize else onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )

                # 2. 양자화 수행 (공통 유틸리티 사용)
                if quantize:
                    if quantizer_tool.quantize_file(str(temp_onnx_path), str(onnx_path)):
                        # 임시 FP32 파일 삭제
                        if os.path.exists(temp_onnx_path):
                            os.remove(temp_onnx_path)
                        print(f"   └ ✨ INT8 양자화 완료: {onnx_path.name}")
                
                converted_files.append(onnx_path)
            except Exception as e:
                print(f"❌ 레벨 {level} 변환 실패: {e}")

        print(f"\n✨ 총 {len(converted_files)}개의 모델이 최적화된 ONNX로 변환되었습니다.")
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
            gcs_service.upload_file(self.bucket_name, str(file_path), blob_name)
            
        print(f"✅ 모든 ONNX 모델이 GCS에 업로드되었습니다.")


if __name__ == "__main__":
    converter = ONNXConverter()
    files = converter.convert_all()
    
    if files:
        confirm = input("\n📤 변환된 파일을 GCS에 업로드하시겠습니까? (y/n): ").lower()
        if confirm == 'y':
            converter.upload_to_gcs(files)
