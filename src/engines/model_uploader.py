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

    def _generate_handler_py(self):
        """
        URL 또는 바이너리 이미지를 모두 처리하는 동적 handler.py를 생성합니다.
        """
        from src.consts.magic_eye_assets import MAGIC_EYE_ASSETS
        categories = [asset["id"] for asset in MAGIC_EYE_ASSETS]
        
        handler_content = f"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import requests

class EndpointHandler():
    def __init__(self, path="."):
        self.path = path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categories = {categories}
        self.models = {{}}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _get_model(self, level):
        if level in self.models:
            return self.models[level]
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, len(self.categories))
        model_path = os.path.join(self.path, "models", f"ai_lv{{level}}.pth")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models[level] = model
            return model
        raise ValueError(f"Model level {{level}} not found")

    def __call__(self, data):
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", {{}})
        level = int(parameters.get("level", 10))

        # 1. 입력이 URL인지 바이너리인지 판단하여 이미지 로드
        try:
            if isinstance(inputs, str) and (inputs.startswith("http://") or inputs.startswith("https://")):
                # URL로부터 이미지 다운로드
                response = requests.get(inputs, timeout=10)
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
            elif isinstance(inputs, bytes):
                image = Image.open(io.BytesIO(inputs)).convert("RGB")
            else:
                # 기타 케이스 (base64 등은 프론트에서 전처리 필요하거나 여기서 추가 구현 가능)
                image = Image.open(io.BytesIO(inputs)).convert("RGB")
        except Exception as e:
            return {{"error": f"Failed to load image: {{str(e)}}"}}

        # 2. 모델 추론
        try:
            model = self._get_model(level)
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
            
            label = self.categories[predicted.item()]
            return [{{
                "label": label, 
                "score": float(confidence.item()),
                "level": level
            }}]
        except Exception as e:
            return {{"error": str(e)}}
"""
        handler_path = self.model_dir / "handler.py"
        with open(handler_path, "w", encoding="utf-8") as f:
            f.write(handler_content.strip())
        return handler_path

    def _generate_requirements_txt(self):
        """
        필요한 패키지에 requests를 추가합니다.
        """
        content = "torch\ntorchvision\nPillow\nrequests\n"
        req_path = self.model_dir / "requirements.txt"
        with open(req_path, "w", encoding="utf-8") as f:
            f.write(content)
        return req_path

    def run_upload(self):
        """
        모델 파일들과 함께 Inference를 위한 handler.py, requirements.txt를 업로드합니다.
        """
        print(f"\n{'=' * 50}")
        print(f"📦 AI 플레이어 모델 및 Inference Handler 업로드 시작")
        print(f"📁 대상 폴더: {self.model_dir}")
        print(f"🌐 레포지토리: {self.repo_id}")
        print(f"{'=' * 50}")

        if not HF_TOKEN:
            print("❌ 오류: HF_TOKEN이 설정되지 않았습니다. .env 파일을 확인해 주세요.")
            return

        if not self.model_dir.exists():
            print(f"❌ 오류: 모델 폴더가 존재하지 않습니다: {self.model_dir}")
            return

        # 1. 도우미 파일 생성 (handler.py, requirements.txt)
        print("📝 추론용 handler.py 및 requirements.txt 생성 중...")
        handler_path = self._generate_handler_py()
        req_path = self._generate_requirements_txt()

        # 2. 모델 파일 필터링 (.pth 파일만)
        model_files = sorted(list(self.model_dir.glob("*.pth")))
        if not model_files:
            print("❌ 업로드할 .pth 모델 파일이 없습니다.")
            return

        print(f"🔍 총 {len(model_files)}개의 모델과 2개의 설정 파일을 발견했습니다.")
        
        # 3. 공통 설정 파일 업로드 (루트에 위치)
        print("🚀 설정 파일 업로드 중 (handler.py, requirements.txt)...")
        self.hf_service.upload_file(str(handler_path), "handler.py", "Add custom inference handler")
        self.hf_service.upload_file(str(req_path), "requirements.txt", "Add requirements for inference")

        # 4. 모델 파일 업로드 (models/ 폴더 내에 위치)
        print(f"🚀 모델 파일 업로드 시작...\n")
        pbar = tqdm(model_files, desc="모델 업로드 진행률", unit="file", leave=True)
        
        for model_path in pbar:
            file_name = model_path.name
            pbar.set_description(f"업로드 중: {file_name}")
            
            self.hf_service.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=f"models/{file_name}",
                commit_message=f"Upload {file_name} (AI Player Level)"
            )
            
        pbar.set_description("✅ 모든 업로드 완료")
        pbar.close()

        # 5. 임시 생성 파일 삭제 (선택 사항)
        if handler_path.exists(): os.remove(handler_path)
        if req_path.exists(): os.remove(req_path)

        print(f"\n✨ 모든 업로드 프로세스가 성공적으로 완료되었습니다.")
        print(f"👉 Hugging Face에서 'Inference Endpoint'를 생성하면 즉시 API로 사용 가능합니다.")
        print(f"👉 확인: https://huggingface.co/{self.repo_id}")


if __name__ == "__main__":
    print("\n💡 업로드할 Hugging Face 레포지토리 ID를 입력하세요 (예: user/magic-eye-ai)")
    target_repo = input("👉 입력: ").strip()
    
    if not target_repo:
        print("❌ 레포지토리 ID가 입력되지 않아 종료합니다.")
        sys.exit(1)

    print("\n🔒 레포지토리를 비공개(Private)로 설정하시겠습니까?")
    is_private_input = input("👉 Y (비공개) / N (공개, 기본값): ").strip().upper()
    is_private = True if is_private_input == 'Y' else False

    uploader = ModelUploader(repo_id=target_repo)
    # 선택한 공개/비공개 설정 반영
    uploader.hf_service.ensure_repo_exists(private=is_private)
    uploader.run_upload()
