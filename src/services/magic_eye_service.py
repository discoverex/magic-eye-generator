import io
import os
import random
import shutil
from typing import Optional

import torch
from transformers import CLIPImageProcessor

from src.config.settings import BASE_DIR
from src.consts.magic_eye_assets import MAGIC_EYE_ASSETS
from src.dtos.generated_image import GeneratedImage, TargetDetails
from src.utils.get_diverse_prompts import PromptAgent
from src.utils.stereogram import create_stereogram


class MagicEyeService:
    def __init__(self):
        # 클래스 초기화 시점이 아니라, 실제 로컬에서 사용할 때만 임포트
        import torch
        from diffusers import StableDiffusionPipeline
        from transformers import pipeline

        # 프로젝트 루트 기준 모델 저장 경로 설정
        self.model_dir = os.path.join(BASE_DIR, "models")
        os.makedirs(self.model_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Stable Diffusion 로컬 로드/다운로드 (이미지 생성용)
        image_processor = CLIPImageProcessor.from_pretrained(
            "Lykon/dreamshaper-8",
            subfolder="feature_extractor"
        )
        # cache_dir를 지정하면 해당 폴더에 모델이 저장됩니다.
        sd_path = os.path.join(self.model_dir, "dreamshaper-8")
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            "Lykon/dreamshaper-8",
            cache_dir=sd_path,
            feature_extractor=image_processor,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            local_files_only=os.path.exists(sd_path)  # 파일이 있으면 로컬에서만 가져옴
        ).to(self.device)
        
        # 하위 tqdm 진행바 비활성화 (상위 진행바에서 한 줄로 통합 관리할 예정)
        self.sd_pipe.set_progress_bar_config(disable=True)

        # 2. Depth Estimation 모델 로드/다운로드 (깊이 맵 추출용)
        depth_path = os.path.join(self.model_dir, "dpt-large")
        
        # transformers 로그 레벨 설정 (DPT 관련 경고 억제)
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()

        self.depth_estimator = pipeline(
            "depth-estimation",
            model="Intel/dpt-large",
            use_safetensors=True,  # 보안 이슈 없는 파일 형식 강제
            model_kwargs={"cache_dir": depth_path},
            device=0 if self.device == "cuda" else -1
        )

        self.prompt_agent = PromptAgent()
        self.assets = MAGIC_EYE_ASSETS

    def _clear_debug_folder(self, folder_path: str):
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path): os.unlink(file_path)
                    elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e: print(f"Delete error: {e}")

    async def generate_specific_game(
            self, asset: dict,
            base_prompt: str,
            seed: Optional[int] = None,
            step_callback: Optional[callable] = None,
            batch_size: int = 1
    ) -> list[GeneratedImage]:
        # 시드 설정 (재현성 또는 랜덤성)
        actual_seed = seed if seed is not None else random.randint(0, 1000000)
        generator = torch.Generator(device=self.device).manual_seed(actual_seed)

        refined_prompt = f"{base_prompt}, high contrast, solid 3d silhouette, isolated on white background"
        negative_prompt = "shadows, shading, blurry, soft edges, realistic, photography, complex texture, gradient"

        # 3. 이미지 생성 (CPU 환경에서는 이 단계가 가장 오래 걸림)
        steps = 20 if self.device == "cuda" else 8  # CPU일 때는 스텝을 과감히 줄임
        
        # 최신 diffusers 콜백 사양 (callback_on_step_end)
        def internal_callback(pipe, step, timestep, callback_kwargs):
            if step_callback:
                step_callback(step, steps)
            return callback_kwargs

        outputs = self.sd_pipe(
            prompt=refined_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            generator=generator,
            num_images_per_prompt=batch_size,  # 👈 여기가 배치의 핵심!
            callback_on_step_end=internal_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        ).images

        results = []
        for raw_image in outputs:
            # 4. Depth Map 추출 (이미지별로 수행)
            depth_result = self.depth_estimator(raw_image)
            depth_map = depth_result["depth"]

            # 5. 매직아이 합성
            magic_eye_img = create_stereogram(depth_map)

            # 6. 결과물 인코딩
            prob_io, ans_io = io.BytesIO(), io.BytesIO()
            magic_eye_img.save(prob_io, format='PNG')
            depth_map.save(ans_io, format='PNG')

            results.append(GeneratedImage(
                problem_image=prob_io.getvalue(),
                answer_image=ans_io.getvalue(),
                target_info=TargetDetails(
                    id=asset['id'],
                    display_name=asset['display_name'],
                    keywords=asset['keywords'],
                )
            ))

        return results