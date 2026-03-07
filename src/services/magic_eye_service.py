import io
import os
import random
import shutil
from datetime import datetime
from typing import Optional

import torch

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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Stable Diffusion 모델 로드 (이미지 생성용)
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        # 2. Depth Estimation 모델 로드 (깊이 맵 추출용)
        self.depth_estimator = pipeline(
            "depth-estimation",
            model="Intel/dpt-large",
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

    async def generate_random_game(self) -> GeneratedImage:
        import torch

        # 1. 랜덤 사물 선택
        target = random.choice(self.assets)

        # 2. 랭체인 에이전트로부터 실시간 프롬프트 1개 획득
        # (만약 비용 절감을 위해 고정 프롬프트를 쓰고 싶다면 target['prompt']를 사용하도록 백업 로직 구성 가능)
        count_limit = random.randint(1, 3)
        print(f"🎯 Selected Target: {target['display_name']} | Count: {count_limit}")
        llm_prompts = await self.prompt_agent.get_diverse_prompts(target['display_name'], count=count_limit)
        base_prompt = llm_prompts[0] if llm_prompts else target['prompt']
        print(f"Base Prompt: {base_prompt}")

        refined_prompt = f"{base_prompt}, high contrast, 3d form"
        negative_prompt = "shadows, shading, grey, gradient, blurry, soft edges, realistic, photography, texture"

        # 3. 이미지 생성 (Random Seed 적용으로 다양성 확보)
        generator = torch.Generator(device=self.device).manual_seed(torch.seed())

        raw_image = self.sd_pipe(
            prompt=refined_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,
            generator=generator
        ).images[0]

        # 4. Depth Map 추출
        depth_result = self.depth_estimator(raw_image)
        depth_map = depth_result["depth"]

        # 5. 매직아이 합성
        magic_eye_img = create_stereogram(depth_map)

        # 6. 결과물 인코딩
        prob_io, ans_io = io.BytesIO(), io.BytesIO()
        magic_eye_img.save(prob_io, format='PNG')
        depth_map.save(ans_io, format='PNG')

        debug_path = "debug_outputs"
        self._clear_debug_folder(debug_path)
        os.makedirs(debug_path, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S")
        magic_eye_img.save(f"{debug_path}/{ts}_{target['id']}_problem.png")
        depth_map.save(f"{debug_path}/{ts}_{target['id']}_answer.png")

        return GeneratedImage(
            problem_image=prob_io.getvalue(),
            answer_image=ans_io.getvalue(),
            target_info=TargetDetails(
                id=target['id'],
                display_name=target['display_name'],
                keywords=target['keywords'],
            )
        )

    async def generate_specific_game(self, asset: dict, base_prompt: str, seed: Optional[int] = None) -> GeneratedImage:

        # 시드 설정 (재현성 또는 랜덤성)
        actual_seed = seed if seed is not None else random.randint(0, 1000000)
        generator = torch.Generator(device=self.device).manual_seed(actual_seed)

        refined_prompt = f"{base_prompt}, high contrast, solid 3d silhouette, isolated on white background"
        negative_prompt = "shadows, shading, blurry, soft edges, realistic, photography, complex texture, gradient"

        # 3. 이미지 생성 (CPU 환경에서는 이 단계가 가장 오래 걸림)
        raw_image = self.sd_pipe(
            prompt=refined_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,  # CPU 속도를 고려해 단계를 약간 줄임
            generator=generator
        ).images[0]

        # 4. Depth Map 추출
        depth_result = self.depth_estimator(raw_image)
        depth_map = depth_result["depth"]

        # 5. 매직아이 합성
        magic_eye_img = create_stereogram(depth_map)

        # 6. 결과물 인코딩 및 반환
        prob_io, ans_io = io.BytesIO(), io.BytesIO()
        magic_eye_img.save(prob_io, format='PNG')
        depth_map.save(ans_io, format='PNG')

        return GeneratedImage(
            problem_image=prob_io.getvalue(),
            answer_image=ans_io.getvalue(),
            target_info=TargetDetails(
                id=asset['id'],
                display_name=asset['display_name'],
                keywords=asset['keywords'],
            )
        )