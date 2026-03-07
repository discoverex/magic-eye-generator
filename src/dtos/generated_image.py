from typing import List

from pydantic import BaseModel, Field


class TargetDetails(BaseModel):
    id: str = Field(..., description="대상 식별자")
    display_name: str = Field(..., description="표시될 이름")
    keywords: List[str] = Field(..., description="키워드 목록")

class GeneratedImage(BaseModel):
    problem_image: bytes = Field(..., description="Base64로 인코딩된 매직아이 문제 이미지")
    answer_image: bytes = Field(..., description="Base64로 인코딩된 정답 깊이 맵 이미지")
    target_info: TargetDetails = Field(..., description="대상 상세 정보")