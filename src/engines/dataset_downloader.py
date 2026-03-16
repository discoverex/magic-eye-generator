import os
from tqdm import tqdm
from src.config.settings import IMAGE_BUCKET_NAME, BASE_DIR
from src.services.gcp_storage_service import GCPStorageService

class DatasetDownloader:
    """
    GCS에 업로드되어 있는 매직아이 데이터셋을 로컬로 내려받는 클래스
    """
    def __init__(self, bucket_name: str = IMAGE_BUCKET_NAME):
        self.bucket_name = bucket_name
        self.storage_service = GCPStorageService()
        self.dataset_dir = os.path.join(BASE_DIR, "datasets")
        self.remote_prefix = "magic-eye/"

    def run(self):
        """
        GCS로부터 데이터셋 다운로드 실행
        """
        print(f"🔍 GCS 버킷 '{self.bucket_name}'에서 데이터셋 목록을 가져오는 중...")
        blobs = self.storage_service.list_blobs(self.bucket_name, prefix=self.remote_prefix)
        
        if not blobs:
            print(f"❌ '{self.remote_prefix}' 경로에 데이터가 없습니다.")
            return

        # metadata.csv가 있는지 확인
        has_metadata = any(blob.name == f"{self.remote_prefix}metadata.csv" for blob in blobs)
        if not has_metadata:
            print("⚠️ 경고: metadata.csv 파일을 찾을 수 없습니다. 데이터셋이 불완전할 수 있습니다.")

        print(f"🚀 다운로드 시작 (총 {len(blobs)}개 파일)")
        
        count_downloaded = 0
        count_failed = 0

        # tqdm 진행바 설정
        progress_bar = tqdm(blobs, desc="데이터셋 다운로드 중", unit="파일", leave=True)

        for blob in progress_bar:
            remote_path = blob.name
            # magic-eye/ -> datasets/ 로 매핑
            relative_path = remote_path[len(self.remote_prefix):]
            if not relative_path: # 폴더 자체인 경우 건너뜀
                continue
                
            local_path = os.path.join(self.dataset_dir, relative_path)
            
            # 진행바에 현재 파일명 표시
            progress_bar.set_postfix(file=os.path.basename(local_path))

            if self.storage_service.download_file(self.bucket_name, remote_path, local_path):
                count_downloaded += 1
            else:
                count_failed += 1

        progress_bar.close()

        print(f"\n✨ 다운로드 완료!")
        print(f"✅ 성공: {count_downloaded}개")
        if count_failed > 0:
            print(f"❌ 실패: {count_failed}개")

if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.run()
