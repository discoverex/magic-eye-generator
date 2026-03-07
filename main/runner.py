import os
import subprocess
import sys
import msvcrt

def run_script(module_path):
    """
    지정한 모듈을 PYTHONPATH=src 환경에서 실행합니다.
    사용자가 Ctrl+C 또는 중단 신호를 보내면 메뉴로 안전하게 복귀합니다.
    """
    env = os.environ.copy()
    # 프로젝트 루트 기준 src 폴더 경로 설정 (main/runner.py 기준 상위 폴더의 src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(project_root, "src")
    env["PYTHONPATH"] = src_dir

    print("\n" + "-"*40)
    print("💡 실행 중 [Ctrl + C]를 누르면 중단하고 메뉴로 돌아갑니다.")
    print("-"*40 + "\n")

    try:
        # subprocess.run은 KeyboardInterrupt를 부모 프로세스로 전달합니다.
        subprocess.run([sys.executable, "-m", module_path], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 실행 중 오류가 발생했습니다 (종료 코드: {e.returncode})")
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 실행이 중단되었습니다. 메뉴로 돌아갑니다.")
    except Exception as e:
        print(f"\n❌ 알 수 없는 오류 발생: {e}")

def get_input_with_esc():
    """
    Esc 키 입력 시 즉시 종료하거나, 숫자를 입력받아 반환합니다.
    """
    print("👉 실행할 작업의 번호를 입력하세요 (종료하려면 'Esc' 또는 '4' 입력): ", end="", flush=True)
    
    input_str = ""
    while True:
        if msvcrt.kbhit():
            char = msvcrt.getch()
            # Esc 키 (27)
            if ord(char) == 27:
                print("\n\n👋 프로그램을 종료합니다.")
                sys.exit(0)
            # 엔터 키 (13)
            elif ord(char) in [13, 10]:
                print()
                return input_str.strip()
            # 백스페이스 (8)
            elif ord(char) == 8:
                if len(input_str) > 0:
                    input_str = input_str[:-1]
                    sys.stdout.write('\b \b')
                    sys.stdout.flush()
            # 일반 문자
            else:
                try:
                    c = char.decode('utf-8')
                    input_str += c
                    sys.stdout.write(c)
                    sys.stdout.flush()
                except:
                    pass

def start_app():
    """
    애플리케이션의 메인 루프를 시작합니다.
    """
    while True:
        print("\n" + "="*45)
        print(" 🌀 StereoVision Showdown: AI vs Human")
        print("="*45)
        print(" 1. 매직아이 데이터셋 생성 (dataset_generator.py)")
        print(" 2. AI 모델 단계별 학습 (trainer.py)")
        print(" 3. GCP Storage에 데이터 업로드 (gcp_storage.py)")
        print(" 4. 종료 (Exit 또는 Esc)")
        print("-" * 45)
        
        choice = get_input_with_esc()

        if choice == "1":
            print("\n🚀 데이터셋 생성을 시작합니다...")
            run_script("core.dataset_generator")
        elif choice == "2":
            print("\n🚀 AI 모델 학습을 시작합니다...")
            run_script("core.trainer")
        elif choice == "3":
            print("\n🚀 GCP Storage 업로드를 시작합니다...")
            run_script("services.gcp_storage")
        elif choice in ["4", "exit", "quit"]:
            print("\n👋 프로그램을 종료합니다. 즐거운 하루 되세요!")
            break
        elif choice == "":
            continue
        else:
            print(f"\n❗ '{choice}'은(는) 잘못된 입력입니다. 1~4번 중에서 선택해주세요.")
