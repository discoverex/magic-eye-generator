import cv2
import numpy as np

def create_simple_depth_map(width, height):
    """테스트용 가짜 깊이 지도 생성 (가운데에 사각형이 튀어나온 형태)"""
    depth_map = np.zeros((height, width), dtype=np.uint8)
    # 중앙에 100x100 크기의 흰색 사각형 (가장 가까운 곳)
    cv2.rectangle(depth_map, (width//2-50, height//2-50), (width//2+50, height//2+50), 255, -1)
    return depth_map