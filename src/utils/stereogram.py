import numpy as np
from PIL import Image


def create_stereogram(depth_map_edge: Image.Image, pattern_div: int = 8):
    """
    Depth Map을 기반으로 매직아이(Autostereogram) 이미지를 생성합니다.
    """
    depth_data = np.array(depth_map_edge.convert("L"))
    width, height = depth_map_edge.size

    # 패턴의 너비 결정 (이미지 너비의 1/8 정도)
    pattern_width = width // pattern_div
    out_data = np.zeros((height, width, 3), dtype=np.uint8)

    # 초기 랜덤 노이즈 패턴 생성
    for y in range(height):
        for x in range(width):
            if x < pattern_width:
                # 왼쪽 첫 구간은 랜덤 색상 채우기
                out_data[y, x] = np.random.randint(0, 255, 3)
            else:
                # 깊이 값에 따른 오프셋 계산 (흰색일수록 더 많이 당겨짐)
                # 0~255 값을 0~20픽셀 정도의 변화로 변환
                shift = int((depth_data[y, x] / 255.0) * (pattern_width * 0.15))
                source_x = x - pattern_width + shift
                out_data[y, x] = out_data[y, source_x]

    return Image.fromarray(out_data)