import numpy as np
from PIL import Image


def create_stereogram(depth_map_edge: Image.Image, pattern_div: int = 8):
    """
    Depth Map을 기반으로 매직아이(Autostereogram) 이미지를 생성합니다.
    NumPy의 인덱싱 기능을 극한으로 활용하여 루프 내 연산을 최소화했습니다.
    """
    # 1. 입력 전처리 (PIL -> NumPy 최적화)
    depth_data = np.asarray(depth_map_edge.convert("L"))
    height, width = depth_data.shape
    pattern_width = width // pattern_div

    # 2. 출력 배열 및 초기 패턴 생성 (RGB)
    out_data = np.empty((height, width, 3), dtype=np.uint8)
    
    # 왼쪽 첫 구간은 랜덤 색상으로 미리 채움
    out_data[:, :pattern_width] = np.random.randint(0, 255, (height, pattern_width, 3), dtype=np.uint8)

    # 3. 오프셋 맵 미리 계산 (루프 밖에서 처리)
    # shift_map: 각 픽셀이 참조할 왼쪽 픽셀의 상대적 오프셋
    shift_map = (depth_data.astype(np.float32) / 255.0 * (pattern_width * 0.15)).astype(np.int32)
    
    # 4. 고정 인덱스 미리 생성 (중요: 루프 밖에서 한 번만 생성)
    rows = np.arange(height)

    # 5. 스테레오그램 합성 (가로 루프는 의존성 때문에 유지하되 내부 연산 극소화)
    for x in range(pattern_width, width):
        # source_x: 이 열이 참조할 이전 열들의 인덱스 (벡터)
        source_indices = x - pattern_width + shift_map[:, x]
        # 고급 인덱싱을 사용하여 모든 행의 x열 데이터를 한 번에 복사
        out_data[:, x] = out_data[rows, source_indices]

    return Image.fromarray(out_data)