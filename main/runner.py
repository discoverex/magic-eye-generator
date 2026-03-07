import os
import subprocess
import sys

def run_script(module_path, *args):
    """
    지정한 모듈을 실행합니다.
    프로젝트 루트를 PYTHONPATH에 추가하여 src. 패키지 경로를 인식하게 합니다.
    """
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # PYTHONPATH에 프로젝트 루트를 추가하여 src.xxx 임포트가 가능하게 함
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = project_root

    print("\n" + "-"*40)
    print("💡 실행 중 [Ctrl + C]를 누르면 중단하고 메뉴로 돌아갑니다.")
    print("💡 파이참 터미널에서 실행중이라면 [Ctrl + F2]를 눌러 중단하세요.")
    print("-"*40 + "\n")

    try:
        # 모듈 경로와 인자들을 합쳐서 실행 명령 구성
        cmd = [sys.executable, "-m", f"src.{module_path}"] + list(args)
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 실행 중 오류가 발생했습니다 (종료 코드: {e.returncode})")
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 실행이 중단되었습니다. 메뉴로 돌아갑니다.")
    except Exception as e:
        print(f"\n❌ 알 수 없는 오류 발생: {e}")

def start_app():
    """
    애플리케이션의 메인 루프를 시작합니다.
    """
    while True:
        print("\n" + "="*45)
        print(" 🌀 StereoVision Showdown: AI vs Human")
        print("="*45)
        print(" 1. 매직아이 데이터셋 초기화 (dataset_initializer.py)")
        print(" 2. 매직아이 데이터셋 생성 (dataset_generator.py)")
        print(" 3. AI 모델 단계별 학습 (model_trainer.py)")
        print(" 4. AI 모델 최종 테스트 (model_tester.py)")
        print(" 5. GCP Storage에 데이터 업로드 (image_uploader.py)")
        print(" 6. AI 모델 Hugging Face 업로드 (model_uploader.py)")
        print(" 7. 종료 (Exit)")
        print("-" * 45)
        
        choice = input("👉 실행할 작업의 번호를 입력하세요 (1~7): ").strip()

        if choice == "1":
            print("\n🧹 데이터셋 초기화를 시작합니다...")
            run_script("engines.dataset_initializer")
        elif choice == "2":
            print("\n🚀 데이터셋 생성을 시작합니다...")
            print("💡 에셋당 생성할 이미지 개수를 입력하세요. (기본값: 300)")
            count_input = input("👉 입력 (엔터 시 300): ").strip()
            
            if count_input == "":
                run_script("engines.dataset_generator")
            else:
                run_script("engines.dataset_generator", count_input)
        elif choice == "3":
            print("\n🚀 AI 모델 학습을 시작합니다...")
            run_script("engines.model_trainer")
        elif choice == "4":
            print("\n🧪 AI 모델 최종 테스트를 시작합니다...")
            run_script("engines.model_tester")
        elif choice == "5":
            print("\n🚀 GCP Storage 업로드를 시작합니다...")
            run_script("engines.image_uploader")
        elif choice == "6":
            print("\n🚀 Hugging Face 업로드를 시작합니다...")
            run_script("engines.model_uploader")
        elif choice in ["7", "exit", "quit"]:
            print("\n👋 프로그램을 종료합니다. 즐거운 하루 되세요!")
            break
        elif choice == "":
            continue
        else:
            print(f"\n❗ '{choice}'은(는) 잘못된 입력입니다. 1~7번 중에서 선택해주세요.")
