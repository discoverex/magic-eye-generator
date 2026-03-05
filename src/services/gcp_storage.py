import os
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

class GCPStorageService:
    def __init__(self):
        # 환경 변수에서 인증 정보를 가져옵니다.
        # 로컬 개발 시에는 GOOGLE_APPLICATION_CREDENTIALS 환경 변수에 JSON 키 경로를 설정해야 합니다.
        self.bucket_name = os.getenv("GCP_BUCKET_NAME")
        self.client = storage.Client()

    def upload_image(self, local_file_path, destination_blob_name):
        """이미지를 GCP 버킷에 업로드하고 공용 URL을 반환합니다."""
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(local_file_path)

        # 나중에 프론트엔드에서 바로 접근 가능하도록 URL 반환
        return blob.public_url