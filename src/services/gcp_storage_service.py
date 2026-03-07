import json
from typing import Optional

from google.cloud import storage
from src.config.settings import GCP_SERVICE_ACCOUNT_JSON


class GCPStorageService:
    """
    Google Cloud Storage 연동을 담당하는 서비스 클래스
    """

    def __init__(self):
        self.client = self._init_client()

    def _init_client(self) -> Optional[storage.Client]:
        """
        환경 변수의 JSON 설정을 바탕으로 GCS 클라이언트를 초기화합니다.
        """
        if not GCP_SERVICE_ACCOUNT_JSON:
            print("❌ 에러: .env 파일에 GCP_SERVICE_ACCOUNT_JSON 설정이 없습니다.")
            return None

        try:
            service_account_info = json.loads(GCP_SERVICE_ACCOUNT_JSON)
            return storage.Client.from_service_account_info(service_account_info)
        except Exception as e:
            print(f"❌ GCP 클라이언트 초기화 에러: {e}")
            return None

    def get_bucket(self, bucket_name: str) -> Optional[storage.Bucket]:
        """
        버킷 객체를 가져옵니다.
        """
        if not self.client:
            return None
        return self.client.bucket(bucket_name)

    def blob_exists(self, bucket_name: str, remote_path: str) -> bool:
        """
        해당 경로에 파일이 이미 존재하는지 확인합니다.
        """
        bucket = self.get_bucket(bucket_name)
        if not bucket:
            return False
        blob = bucket.blob(remote_path)
        return blob.exists()

    def upload_file(self, bucket_name: str, local_path: str, remote_path: str) -> bool:
        """
        로컬 파일을 GCS에 업로드합니다.
        """
        bucket = self.get_bucket(bucket_name)
        if not bucket:
            return False

        try:
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            return True
        except Exception as e:
            print(f"\n❌ 업로드 실패 ({remote_path}): {e}")
            return False
