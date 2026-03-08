import sys

from tqdm import tqdm as tqdm_cls


def configure_tqdm():
    """
    tqdm의 기본 출력을 stderr에서 stdout으로 강제 변경합니다.
    (PyCharm 등에서 stderr가 붉은색으로 표시되는 현상 해결용)
    """
    # 1. 이미 패치되었는지 확인
    if hasattr(tqdm_cls, '_is_patched_for_stdout'):
        return
        
    # 2. tqdm 클래스의 __init__ 기본 인자를 변경 시도
    orig_init = tqdm_cls.__init__
    
    def patched_init(self, *args, **kwargs):
        # file 인자가 명시되지 않았다면 기본값으로 stdout을 사용
        if 'file' not in kwargs:
            kwargs['file'] = sys.stdout
        orig_init(self, *args, **kwargs)
        
    tqdm_cls.__init__ = patched_init
    tqdm_cls._is_patched_for_stdout = True
    
    # 3. 전역 tqdm 모듈의 인스턴스들도 패치된 클래스를 가리키도록 설정
    import tqdm
    tqdm.tqdm = tqdm_cls
