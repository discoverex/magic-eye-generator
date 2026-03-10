import os
from src.config.settings import BASE_DIR, BUCKET_NAME
from src.services.gcp_storage_service import GCPStorageService

def upload_test_results():
    """
    test_results 폴더 내의 모든 파일(.png, .json)을 GCP Storage의 'test_results' 폴더로 업로드합니다.
    """
    result_dir = os.path.join(BASE_DIR, "test_results")
    if not os.path.exists(result_dir):
        print(f"❌ 업로드할 결과 폴더가 존재하지 않습니다: {result_dir}")
        return

    files = [f for f in os.listdir(result_dir) if os.path.isfile(os.path.join(result_dir, f))]
    if not files:
        print("📁 업로드할 파일이 없습니다.")
        return

    print(f"🚀 총 {len(files)}개의 결과 파일을 업로드합니다...")
    
    storage_service = GCPStorageService()
    
    success_count = 0
    for filename in files:
        local_path = os.path.join(result_dir, filename)
        # GCS 내 저장 경로: test_results/파일명
        blob_name = f"test_results/{filename}"
        
        if storage_service.upload_file(BUCKET_NAME, local_path, blob_name):
            print(f"  ✅ 업로드 성공: {filename}")
            success_count += 1
        else:
            print(f"  ❌ 업로드 실패: {filename}")

    print(f"\n✨ 업로드 완료! (성공: {success_count}/{len(files)})")

if __name__ == "__main__":
    upload_test_results()
