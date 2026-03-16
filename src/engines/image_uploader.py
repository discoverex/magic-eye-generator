import os

import pandas as pd
from tqdm import tqdm

from src.config.settings import BUCKET_NAME, BASE_DIR
from src.services.gcp_storage_service import GCPStorageService


class ImageUploader:
    """
    로컬의 매직아이 데이터셋 중 테스트용(split='test') 데이터만 선별하여 GCS로 업로드하는 클래스
    """

    def __init__(self, bucket_name: str = BUCKET_NAME):
        self.bucket_name = bucket_name
        self.storage_service = GCPStorageService()
        self.dataset_dir = os.path.join(BASE_DIR, "datasets")
        self.metadata_path = os.path.join(self.dataset_dir, "metadata.csv")

    def run(self, overwrite: bool = True):
        """
        업로드 로직 통합 실행기 (전체 이미지 + 메타데이터)
        """
        if not os.path.exists(self.metadata_path):
            print(f"❌ 에러: 메타데이터 파일이 없습니다. ({self.metadata_path})")
            return

        print(f"🔍 메타데이터 분석 중: {self.metadata_path}")
        
        # 1. 전체 데이터 로드 (필터링 제거)
        try:
            df = pd.read_csv(self.metadata_path)
        except Exception as e:
            print(f"❌ CSV 읽기 에러: {e}")
            return

        if df.empty:
            print("📁 업로드할 데이터가 없습니다. 먼저 데이터셋을 생성해 주세요.")
            return

        # 2. 업로드 대상 태스크 구성 (문제 이미지와 정답 이미지 모두 포함)
        upload_tasks = self._collect_tasks_from_df(df)

        # 3. 메타데이터 파일 추가 (GCS의 magic-eye/metadata.csv 경로로 업로드)
        remote_metadata_path = "magic-eye/metadata.csv"
        upload_tasks.append((self.metadata_path, remote_metadata_path))

        print(f"🚀 GCS 업로드 시작 (대상: 전체 데이터 {len(df)}세트 + 메타데이터, 총 {len(upload_tasks)}개 파일)")
        print(f"📂 버킷 경로: gs://{self.bucket_name}/magic-eye/...")
        
        if overwrite:
            print("⚠️ 모드: 덮어쓰기 (기존 파일이 있으면 대체합니다.)")
        else:
            print("🛡️ 모드: 건너뛰기 (이미 존재하는 파일은 업로드하지 않습니다.)")

        # 4. 업로드 실행
        self._execute_uploads(upload_tasks, overwrite=overwrite)

    def run_metadata_only(self, overwrite: bool = True):
        """
        metadata.csv 파일만 선별하여 GCS로 업로드
        """
        if not os.path.exists(self.metadata_path):
            print(f"❌ 에러: 메타데이터 파일이 없습니다. ({self.metadata_path})")
            return

        remote_metadata_path = "magic-eye/metadata.csv"
        
        if not overwrite and self.storage_service.blob_exists(self.bucket_name, remote_metadata_path):
            print(f"⏭️ 건너뜀: 메타데이터가 이미 존재합니다. ({remote_metadata_path})")
            return

        print(f"📝 메타데이터 단독 업로드 중: {remote_metadata_path}")
        
        success = self.storage_service.upload_file(self.bucket_name, self.metadata_path, remote_metadata_path)
        if success:
            print(f"✅ 메타데이터 업로드 완료!")
        else:
            print(f"❌ 메타데이터 업로드 실패!")

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

    def _execute_uploads(self, tasks: list, overwrite: bool = True):
        """
        수집된 업로드 태스크를 수행합니다.
        """
        count_uploaded = 0
        count_skipped = 0
        count_failed = 0

        # tqdm 진행바 설정
        progress_bar = tqdm(tasks, desc="데이터 업로드 중", unit="파일", leave=True)

        for local_path, remote_path in progress_bar:
            # 현재 파일명을 진행바에 표시
            file_name = os.path.basename(local_path)
            progress_bar.set_postfix(file=file_name)

            # 덮어쓰지 않을 경우 존재 여부 확인
            if not overwrite:
                if self.storage_service.blob_exists(self.bucket_name, remote_path):
                    count_skipped += 1
                    continue

            # 업로드 수행
            try:
                if self.storage_service.upload_file(self.bucket_name, local_path, remote_path):
                    count_uploaded += 1
                else:
                    count_failed += 1
            except Exception as e:
                count_failed += 1
                progress_bar.write(f"❌ 업로드 중 예상치 못한 에러 ({file_name}): {e}")

        progress_bar.close()

        print(f"\n✨ 업로드 프로세스 완료!")
        print(f"✅ 신규 업로드: {count_uploaded}개")
        if not overwrite:
            print(f"⏭️ 건너뜀 (이미 존재): {count_skipped}개")
        if count_failed > 0:
            print(f"❌ 실패: {count_failed}개")


if __name__ == "__main__":
    import sys
    
    # 인자 처리: python -m src.engines.image_uploader [overwrite: True/False]
    overwrite_arg = True
    if len(sys.argv) > 1:
        # "false" 또는 "0"인 경우 False로 설정
        val = sys.argv[1].lower()
        if val in ["false", "0", "no", "n"]:
            overwrite_arg = False
            
    uploader = ImageUploader()
    uploader.run(overwrite=overwrite_arg)
