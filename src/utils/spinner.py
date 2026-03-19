import sys
import time
import threading
from typing import Optional


class Spinner:
    """
    작업 대기 중 터미널에 스피너를 표시하는 컨텍스트 매니저 클래스
    """
    def __init__(self, message: str = "처리 중...", delay: float = 0.1):
        self.message = message
        self.delay = delay
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def _spin(self):
        idx = 0
        while not self.stop_event.is_set():
            # \r (Carriage Return)으로 줄 시작 위치로 이동 후 덮어쓰기
            sys.stdout.write(f"\r {self.spinner_chars[idx % len(self.spinner_chars)]} {self.message}")
            sys.stdout.flush()
            idx += 1
            time.sleep(self.delay)
        
        # 종료 시 현재 줄 지우기
        sys.stdout.write("\r" + " " * (len(self.message) + 5) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True # 메인 프로세스 종료 시 함께 종료
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.thread:
            self.stop_event.set()
            self.thread.join()
