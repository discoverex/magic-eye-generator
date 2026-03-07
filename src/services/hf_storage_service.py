import os
from typing import Optional

from huggingface_hub import HfApi, create_repo

from src.config.settings import HF_TOKEN


class HFStorageService:
    """
    Hugging Face Hub와 통신하여 모델 파일을 업로드하고 관리하는 서비스 클래스
    """
    def __init__(self, repo_id: str, token: Optional[str] = HF_TOKEN):
        self.api = HfApi(token=token)
        self.repo_id = repo_id
        self.token = token

    def ensure_repo_exists(self, repo_type: str = "model", private: bool = False):
        """
        레포지토리가 존재하지 않으면 생성합니다.
        """
        try:
            create_repo(
                repo_id=self.repo_id,
                token=self.token,
                repo_type=repo_type,
                private=private,
                exist_ok=True
            )
            print(f"✅ Hugging Face 레포지토리 준비 완료: {self.repo_id}")
        except Exception as e:
            print(f"⚠️ 레포지토리 생성 중 오류 발생 (이미 존재할 수 있음): {e}")

    def upload_file(self, path_or_fileobj: str, path_in_repo: str, commit_message: Optional[str] = None):
        """
        단일 파일을 레포지토리에 업로드합니다.
        """
        if not os.path.exists(path_or_fileobj):
            print(f"❌ 업로드할 파일을 찾을 수 없습니다: {path_or_fileobj}")
            return

        try:
            self.api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                commit_message=commit_message or f"Upload {path_in_repo}",
                token=self.token
            )
            print(f"🚀 업로드 성공: {path_in_repo}")
        except Exception as e:
            print(f"❌ 업로드 중 오류 발생 ({path_in_repo}): {e}")

    def upload_folder(self, folder_path: str, path_in_repo: str = "models", commit_message: Optional[str] = None):
        """
        폴더 전체를 레포지토리에 업로드합니다.
        """
        if not os.path.isdir(folder_path):
            print(f"❌ 업로드할 폴더를 찾을 수 없습니다: {folder_path}")
            return

        try:
            self.api.upload_folder(
                folder_path=folder_path,
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                commit_message=commit_message or f"Upload folder {path_in_repo}",
                token=self.token
            )
            print(f"🚀 폴더 전체 업로드 성공: {folder_path} -> {path_in_repo}")
        except Exception as e:
            print(f"❌ 폴더 업로드 중 오류 발생: {e}")
