import os
import sys
from pathlib import Path
from tqdm import tqdm

from src.config.settings import BASE_DIR, HF_TOKEN
from src.services.hf_storage_service import HFStorageService


class ModelUploader:
    """
    훈련된 10단계 AI 모델들을 Hugging Face에 업로드하는 엔진 클래스 (tqdm 진행바 포함)
    """
    def __init__(self, repo_id: str):
        self.model_dir = BASE_DIR / "models" / "players"
        self.repo_id = repo_id
        self.hf_service = HFStorageService(repo_id=repo_id)

    def run_upload(self):
        """
        모델 폴더 내의 모든 .pth 파일을 허깅페이스로 업로드합니다.
        tqdm을 사용하여 실시간 진행 상태를 표시합니다.
        """
        print(f"\n{'=' * 50}")
        print(f"📦 AI 플레이어 모델 허깅페이스 업로드 시작")
        print(f"📁 대상 폴더: {self.model_dir}")
        print(f"🌐 레포지토리: {self.repo_id}")
        print(f"{'=' * 50}")

        if not HF_TOKEN:
            print("❌ 오류: HF_TOKEN이 설정되지 않았습니다. .env 파일을 확인해 주세요.")
            return

        if not self.model_dir.exists():
            print(f"❌ 오류: 모델 폴더가 존재하지 않습니다: {self.model_dir}")
            return

        # 1. 레포지토리 준비
        self.hf_service.ensure_repo_exists(private=False)

        # 2. 모델 파일 필터링 (.pth 파일만)
        model_files = sorted(list(self.model_dir.glob("*.pth")))
        if not model_files:
            print("❌ 업로드할 .pth 모델 파일이 없습니다.")
            return

        print(f"🔍 총 {len(model_files)}개의 모델 파일을 발견했습니다.")
        print(f"🚀 실시간 업로드를 시작합니다...\n")

        # 3. tqdm 진행바와 함께 개별 파일 업로드
        # 개별 파일 업로드 시 Hugging Face Hub 라이브러리 자체에서도 내부 진행바를 표시할 수 있으나,
        # 여기서는 전체 파일 수에 대한 진행바를 명시적으로 구현합니다.
        pbar = tqdm(model_files, desc="전체 진행률", unit="file", leave=True)
        
        for model_path in pbar:
            file_name = model_path.name
            pbar.set_description(f"업로드 중: {file_name}")
            
            # 개별 파일 업로드 수행
            self.hf_service.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=f"models/{file_name}",
                commit_message=f"Upload {file_name} (AI Player Level)"
            )
            
        pbar.set_description("✅ 모든 업로드 완료")
        pbar.close()

        print(f"\n✨ 모든 모델 업로드 프로세스가 성공적으로 완료되었습니다.")
        print(f"👉 확인: https://huggingface.co/{self.repo_id}/tree/main/models")


if __name__ == "__main__":
    print("\n💡 업로드할 Hugging Face 레포지토리 ID를 입력하세요 (예: user/magic-eye-ai)")
    target_repo = input("👉 입력: ").strip()
    
    if not target_repo:
        print("❌ 레포지토리 ID가 입력되지 않아 종료합니다.")
        sys.exit(1)

    uploader = ModelUploader(repo_id=target_repo)
    uploader.run_upload()
