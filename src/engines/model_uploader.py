import os
from pathlib import Path
from tqdm import tqdm

from src.config.settings import BASE_DIR, BUCKET_NAME
from src.services.gcp_storage_service import GCPStorageService


class ModelUploader:
    """
    변환된 ONNX 모델들을 GCP Storage(GCS)의 'models/onnx/' 경로로 업로드하는 엔진 클래스.
    """
    def __init__(self, bucket_name: str = BUCKET_NAME):
        self.model_dir = BASE_DIR / "models" / "onnx"
        self.bucket_name = bucket_name
        self.storage_service = GCPStorageService()

    def run(self, overwrite: bool = True):
        """
        ONNX 모델 업로드 프로세스를 실행합니다.
        """
        if not self.model_dir.exists():
            print(f"❌ 오류: ONNX 모델 폴더가 존재하지 않습니다: {self.model_dir}")
            return

        # 업로드 대상 파일 수집 (.onnx)
        model_files = sorted(list(self.model_dir.glob("*.onnx")))
        if not model_files:
            print("❌ 업로드할 .onnx 모델 파일이 없습니다. 먼저 ONNX 변환을 수행해 주세요.")
            return

        print(f"\n{'=' * 50}")
        print(f"🚀 AI 모델 GCS 업로드 시작 (대상: {len(model_files)}개)")
        print(f"📂 버킷 경로: gs://{self.bucket_name}/models/onnx/...")
        
        if overwrite:
            print("⚠️ 모드: 덮어쓰기 (기존 모델이 있으면 대체합니다.)")
        else:
            print("🛡️ 모드: 건너뛰기 (이미 존재하는 모델은 업로드하지 않습니다.)")
        print(f"{'=' * 50}\n")

        count_uploaded = 0
        count_skipped = 0
        count_failed = 0

        pbar = tqdm(model_files, desc="모델 업로드 중", unit="file")
        
        for local_path in pbar:
            file_name = local_path.name
            remote_path = f"models/onnx/{file_name}"
            pbar.set_postfix(file=file_name)

            # 덮어쓰지 않을 경우 존재 여부 확인
            if not overwrite:
                if self.storage_service.blob_exists(self.bucket_name, remote_path):
                    count_skipped += 1
                    continue

            # 업로드 수행
            try:
                if self.storage_service.upload_file(self.bucket_name, str(local_path), remote_path):
                    count_uploaded += 1
                else:
                    count_failed += 1
            except Exception as e:
                count_failed += 1
                pbar.write(f"❌ 업로드 실패 ({file_name}): {e}")

        pbar.close()

        print(f"\n✨ 모델 업로드 완료!")
        print(f"✅ 신규 업로드: {count_uploaded}개")
        if not overwrite:
            print(f"⏭️ 건너뜀 (이미 존재): {count_skipped}개")
        if count_failed > 0:
            print(f"❌ 실패: {count_failed}개")


if __name__ == "__main__":
    import sys
    
    # 인자 처리: python -m src.engines.model_uploader [overwrite: True/False]
    overwrite_arg = True
    if len(sys.argv) > 1:
        # "false" 또는 "0"인 경우 False로 설정
        val = sys.argv[1].lower()
        if val in ["false", "0", "no", "n"]:
            overwrite_arg = False
            
    uploader = ModelUploader()
    uploader.run(overwrite=overwrite_arg)
