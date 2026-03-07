import os
import pandas as pd
from tqdm import tqdm

from src.services.gcp_storage_service import GCPStorageService
from src.config.settings import BUCKET_NAME, BASE_DIR


class ImageUploader:
    """
    로컬의 매직아이 데이터셋 중 테스트용(split='test') 데이터만 선별하여 GCS로 업로드하는 클래스
    """

    def __init__(self, bucket_name: str = BUCKET_NAME):
        self.bucket_name = bucket_name
        self.storage_service = GCPStorageService()
        self.dataset_dir = os.path.join(BASE_DIR, "datasets")
        self.metadata_path = os.path.join(self.dataset_dir, "metadata.csv")

    def run(self):
        """
        업로드 로직 통합 실행기
        """
        if not os.path.exists(self.metadata_path):
            print(f"❌ 에러: 메타데이터 파일이 없습니다. ({self.metadata_path})")
            return

        print(f"🔍 메타데이터 분석 중: {self.metadata_path}")
        
        # 1. 테스트 데이터 필터링
        try:
            df = pd.read_csv(self.metadata_path)
            test_df = df[df['split'] == 'test']
        except Exception as e:
            print(f"❌ CSV 읽기 에러: {e}")
            return

        if test_df.empty:
            print("📁 업로드할 'test' split 데이터가 없습니다. 먼저 데이터셋을 생성해 주세요.")
            return

        # 2. 업로드 대상 태스크 구성 (문제 이미지와 정답 이미지 모두 포함)
        upload_tasks = self._collect_tasks_from_df(test_df)

        print(f"🚀 GCS 업로드 시작 (대상: TEST 데이터 {len(test_df)}세트, 총 {len(upload_tasks)}개 파일)")
        print(f"📂 버킷 경로: gs://{self.bucket_name}/magic-eye/...")

        # 3. 업로드 실행
        self._execute_uploads(upload_tasks)

    def _collect_tasks_from_df(self, df: pd.DataFrame) -> list:
        """
        데이터프레임의 경로 정보를 바탕으로 업로드 태스크 리스트를 만듭니다.
        """
        tasks = []
        for _, row in df.iterrows():
            # 문제 이미지와 정답 이미지 경로 추출
            for col in ['problem_path', 'answer_path']:
                rel_path = row[col].replace("\\", "/") # 윈도우 경로 호환성
                local_path = os.path.join(self.dataset_dir, rel_path)
                
                # 버킷 내 경로 설정 (magic-eye 폴더 기준)
                remote_path = f"magic-eye/{rel_path}"
                
                if os.path.exists(local_path):
                    tasks.append((local_path, remote_path))
                else:
                    print(f"⚠️ 파일 없음 건너뜀: {local_path}")
        
        return tasks

    def _execute_uploads(self, tasks: list):
        """
        수집된 업로드 태스크를 수행합니다.
        """
        count_uploaded = 0
        count_skipped = 0

        for local_path, remote_path in tqdm(tasks, desc="Uploading Test Data"):
            # 1. 원격지 존재 여부 확인
            if self.storage_service.blob_exists(self.bucket_name, remote_path):
                count_skipped += 1
                continue

            # 2. 업로드 수행
            if self.storage_service.upload_file(self.bucket_name, local_path, remote_path):
                count_uploaded += 1

        print(f"\n✨ 테스트 데이터 업로드 완료!")
        print(f"✅ 신규 업로드: {count_uploaded}개")
        print(f"⏭️ 중복 건너뜀: {count_skipped}개")


if __name__ == "__main__":
    uploader = ImageUploader()
    uploader.run()
