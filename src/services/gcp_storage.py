import asyncio
import json
import os

from dotenv import load_dotenv
from google.cloud import storage
from tqdm import tqdm

from config.settings import GCP_SERVICE_ACCOUNT_JSON, BUCKET_NAME

# .env 파일 로드
load_dotenv()

async def upload_outputs_to_gcs(bucket_name: str):
    # 1. 환경변수에서 JSON 로드
    gcp_json = GCP_SERVICE_ACCOUNT_JSON
    if not gcp_json:
        print("❌ 에러: .env 파일에 GCP_SERVICE_ACCOUNT_JSON 설정이 없습니다.")
        return

    # 2. GCS 클라이언트 초기화
    try:
        service_account_info = json.loads(gcp_json)
        storage_client = storage.Client.from_service_account_info(service_account_info)
    except Exception as e:
        print(f"❌ JSON 파싱 에러: {e}")
        return

    bucket = storage_client.bucket(bucket_name)

    # 로컬 소스 디렉토리 (backend/src/outputs)
    current_file_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))))
    source_dir = os.path.join(base_dir, "src", "outputs")
    print(f"🔍 대상 파일 경로: {source_dir}")

    # 3. 업로드 대상 파일 수집
    upload_tasks = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".png"):
                local_path = os.path.join(root, file)

                # 로컬의 상대 경로 (예: dinosaur/dinosaur_1_prob.png)
                relative_path = os.path.relpath(local_path, source_dir).replace("\\", "/")

                # 버킷 내 최종 경로: magic-eye 폴더 아래에 배치
                remote_path = f"magic-eye/{relative_path}"

                upload_tasks.append((local_path, remote_path))

    if not upload_tasks:
        print("📁 업로드할 PNG 파일이 없습니다. 경로를 확인해주세요.")
        return

    # 4. 업로드 실행 (중복 체크 포함)
    print(f"🚀 GCS 업로드 시작 (버킷: {bucket_name})")
    print(f"📂 대상 경로: gs://{bucket_name}/magic-eye/...")

    count_uploaded = 0
    count_skipped = 0

    for local_path, remote_path in tqdm(upload_tasks, desc="Uploading"):
        blob = bucket.blob(remote_path)

        # [1번 방식] 파일 존재 여부 체크
        # 버킷에 해당 경로의 파일이 이미 있다면 건너뜁니다.
        if blob.exists():
            count_skipped += 1
            continue

        try:
            blob.upload_from_filename(local_path)
            count_uploaded += 1
        except Exception as e:
            print(f"\n❌ 업로드 실패 ({remote_path}): {e}")

    print(f"\n✨ 작업 완료!")
    print(f"✅ 새로 업로드됨: {count_uploaded}개")
    print(f"⏭️ 건너뜀 (이미 존재): {count_skipped}개")


if __name__ == "__main__":
    # 생성하신 실제 버킷 이름을 여기에 입력하세요
    asyncio.run(upload_outputs_to_gcs(BUCKET_NAME))