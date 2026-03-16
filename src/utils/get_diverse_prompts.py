from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List

from src.config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL


class PromptAgent:
    def __init__(self):
        # Ollama 로컬 모델로 교체
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=1.2
        )

        self.template = ChatPromptTemplate.from_messages([
            ("system",
             "You are a creative prompt engineer. You strictly follow the output format and generate high-contrast 3D scene descriptions for image generation."),
            ("user", """
                    [Role]
                    당신은 입력받은 '{target_word}'를 영문으로 번역하고, {count}개의 객체가 포함된 Stable Diffusion용 장면 묘사 프롬프트 1개를 작성합니다.

                    [Task]
                    1. '{target_word}'를 영문으로 번역하세요.
                    2. {count}개의 객체가 화면에 겹치지 않게 배치된 장면을 묘사하세요.
                    3. 각 객체는 서로 다른 각도(Bird's-eye, Side, 45-degree, Frontal)와 정적 상태(Floating, Upside-down, Tilted, Mirrored, Static Pose)를 가져야 합니다.

                    [Mandatory Style Guidelines]
                    - Style: 3D smooth white clay models, high contrast, sharp edges.
                    - Environment: Solid pure black background, floating in a void, no ground, no shadows.
                    - Layout: If count > 1, scatter them separately (e.g., top-left, center, bottom-right) and fill the frame generously.
                    - **Strict Output Rule**: Output ONLY the English prompt string. No conversational text, no quotes, no explanations.

                    [Examples]
                    - Input: target_word='고래', count=1
                      Output: A 3D smooth white clay model of a majestic whale swimming in a void, side view, pure white on a solid black background, high contrast, sharp edges.

                    - Input: target_word='나비', count=3
                      Output: Three 3D smooth white clay models of butterflies scattered in a void: one wing-spread butterfly at top-left, one tilting butterfly in the center, and one side-view butterfly at bottom-right, all in pure white on a solid black background, high contrast, sharp edges.

                    Input: target_word='{target_word}', count={count}
                    Output:""")
        ])

        self.chain = self.template | self.llm | StrOutputParser()

    async def get_diverse_prompts(self, target_word: str, count: int = 1) -> List[str]:
        """
        target_word에 대해 {count}개의 객체가 포함된 단일 프롬프트를 생성합니다.
        결과값은 기존 인터페이스 유지를 위해 List[str]로 반환합니다.
        """
        response = await self.chain.ainvoke({
            "target_word": target_word,
            "count": count
        })

        # 결과값에서 불필요한 따옴표나 사족 제거 후 리스트화
        final_prompt = response.strip().replace('"', '').replace("'", "")
        return [final_prompt]
