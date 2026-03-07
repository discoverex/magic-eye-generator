import sys
from main import start_app

if __name__ == "__main__":
    try:
        start_app()
    except KeyboardInterrupt:
        print("\n\n👋 프로그램을 종료합니다.")
        sys.exit(0)
