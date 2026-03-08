from src.utils.tqdm_config import configure_tqdm

# src.engines 패키지의 어떤 모듈이 로드되더라도 
# tqdm 로그가 PyCharm에서 붉은색으로 보이지 않게 stdout으로 리다이렉션합니다.
configure_tqdm()
