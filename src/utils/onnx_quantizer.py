import os
from pathlib import Path
from typing import Optional

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

from src.config.settings import BASE_DIR


class ONNXQuantizer:
    """
    ONNX 모델을 INT8로 양자화하는 유틸리티 클래스.
    양자화 여부를 확인하고 미양자화 모델만 처리할 수 있습니다.
    """
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir if model_dir else BASE_DIR / "models" / "onnx"

    @staticmethod
    def is_quantized(model_path: str) -> bool:
        """
        모델의 노드 타입을 분석하여 이미 양자화된 모델인지 확인합니다.
        (단순 체크: QuantizeLinear 또는 DequantizeLinear 노드 포함 여부)
        """
        try:
            model = onnx.load(model_path)
            for node in model.graph.node:
                if node.op_type in ["QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"]:
                    return True
            return False
        except Exception:
            return False

    def quantize_file(self, input_path: str, output_path: str) -> bool:
        """
        단일 ONNX 파일을 INT8로 양자화합니다.
        """
        try:
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=QuantType.QUInt8
            )
            return True
        except Exception as e:
            print(f"❌ 양자화 중 오류 발생 ({input_path}): {e}")
            return False

    def quantize_all_in_dir(self):
        """
        지정된 디렉토리 내의 모든 ONNX 모델을 검사하여 미양자화 모델만 양자화합니다.
        """
        if not self.model_dir.exists():
            print(f"📁 디렉토리가 존재하지 않습니다: {self.model_dir}")
            return

        print(f"\n🔍 ONNX 양자화 검사 시작: {self.model_dir}")
        onnx_files = list(self.model_dir.glob("*.onnx"))
        
        if not onnx_files:
            print("📭 양자화할 ONNX 파일이 없습니다.")
            return

        processed_count = 0
        skipped_count = 0

        for model_path in onnx_files:
            # 이미 양자화된 모델인지 확인
            if self.is_quantized(str(model_path)):
                skipped_count += 1
                continue

            print(f"📦 양자화 진행 중: {model_path.name}")
            
            # 임시 파일로 변환 후 덮어쓰기
            temp_output = str(model_path).replace(".onnx", "_quantized.onnx")
            
            if self.quantize_file(str(model_path), temp_output):
                # 원본 삭제 및 이름 변경 (덮어쓰기)
                os.remove(model_path)
                os.rename(temp_output, model_path)
                processed_count += 1
                print(f"   └ ✨ 완료: {model_path.name}")
            else:
                if os.path.exists(temp_output):
                    os.remove(temp_output)

        print(f"\n✅ 양자화 작업 완료!")
        print(f"   - 신규 양자화: {processed_count}개")
        print(f"   - 이미 양자화됨(건너뜀): {skipped_count}개")


if __name__ == "__main__":
    quantizer = ONNXQuantizer()
    quantizer.quantize_all_in_dir()
