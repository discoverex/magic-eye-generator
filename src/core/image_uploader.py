import os
from tqdm import tqdm

from src.services.gcp_storage_service import GCPStorageService
from src.config.settings import BUCKET_NAME, BASE_DIR


class ImageUploader:
    """
    로컬의 매직아이 이미지 데이터를 GCS로 업로드하는 로직을 수행하는 클래스
    """

    def __init__(self, bucket_name: str = BUCKET_NAME):
        self.bucket_name = bucket_name
        self.storage_service = GCPStorageService()
        self.source_dir = os.path.join(BASE_DIR, "datasets")

    def run(self):
        """
        업로드 로직 통합 실행기
        """
        print(f"🔍 대상 로컬 경로: {self.source_dir}")

        # 1. 업로드 대상 파일 수집
        upload_tasks = self._collect_upload_tasks()

        if not upload_tasks:
            print("📁 업로드할 PNG 파일이 없습니다. 경로를 확인해주세요.")
            return

        # 2. 업로드 실행
        print(f"🚀 GCS 업로드 시작 (버킷: {self.bucket_name})")
        print(f"📂 대상 경로: gs://{self.bucket_name}/magic-eye/...")

        self._execute_uploads(upload_tasks)

    def _collect_upload_tasks(self) -> list:
        """
        업로드할 로컬 파일과 원격 경로 쌍을 리스트로 수집합니다.
        """
        tasks = []
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith(".png"):
                    local_path = os.path.join(root, file)

                    # 로컬의 상대 경로 (예: dinosaur/dinosaur_1_prob.png)
                    relative_path = os.path.relpath(local_path, self.source_dir).replace("\\", "/")

                    # 버킷 내 최종 경로: magic-eye 폴더 아래에 배치
                    remote_path = f"magic-eye/{relative_path}"

                    tasks.append((local_path, remote_path))
        return tasks

    def _execute_uploads(self, tasks: list):
        """
        수집된 업로드 태스크를 수행합니다.
        """
        count_uploaded = 0
        count_skipped = 0

        for local_path, remote_path in tqdm(tasks, desc="Uploading"):
            # 1. 원격지 존재 여부 확인
            if self.storage_service.blob_exists(self.bucket_name, remote_path):
                count_skipped += 1
                continue

            # 2. 업로드 수행
            if self.storage_service.upload_file(self.bucket_name, local_path, remote_path):
                count_uploaded += 1

        print(f"\n✨ 작업 완료!")
        print(f"✅ 새로 업로드됨: {count_uploaded}개")
        print(f"⏭️ 건너뜀 (이미 존재): {count_skipped}개")


if __name__ == "__main__":
    uploader = ImageUploader()
    uploader.run()
