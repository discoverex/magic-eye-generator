import os
from pathlib import Path
from typing import Union


def to_cross_platform_path(path: Union[str, Path]) -> str:
    """
    모든 경로 구분자를 '/' (Forward Slash)로 통일하여 반환합니다.
    Windows에서 생성된 백슬래시(\) 경로를 Linux/macOS에서도 호환 가능하도록 변환합니다.
    """
    if path is None:
        return ""
    return str(path).replace('\\', '/')


def join_path(*args) -> str:
    """
    여러 경로 조각을 결합하고, 결과의 구분자를 '/'로 통일하여 반환합니다.
    """
    joined = os.path.join(*args)
    return to_cross_platform_path(joined)


def ensure_local_path(path: Union[str, Path]) -> str:
    """
    현재 운영체제(OS)에 맞는 경로 구분자를 사용하는 경로로 변환합니다.
    이미지 로드 등 로컬 파일 시스템 접근 시 사용합니다.
    """
    if path is None:
        return ""
    # os.path.normpath는 OS에 맞게 구분자를 변경해줌 (\ -> / on Linux, / -> \ on Windows)
    return os.path.normpath(str(path))
