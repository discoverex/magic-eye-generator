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
        print("\n" + "="*50)
        print(" 🌀 StereoVision Showdown: AI vs Human")
        print("="*50)
        print(" [Engines]")
        print(" 1. 매직아이 데이터셋 초기화 (dataset_initializer.py)")
        print(" 2. 매직아이 데이터셋 생성 (dataset_generator.py)")
        print(" 3. AI 모델 단계별 학습 (model_trainer.py)")
        print(" 4. AI 모델 최종 테스트 (model_tester.py)")
        print(" 5. GCP Storage에 데이터 업로드 (image_uploader.py)")
        print(" 6. GCP Storage에 AI 모델 업로드 (model_uploader.py)")
        print(" 7. GCP Storage에서 데이터셋 다운로드 (dataset_downloader.py)")
        print(" 8. GCP Storage에서 AI 모델 다운로드 (model_downloader.py)")
        print("-" * 50)
        print(" [Utilities]")
        print(" 9. 데이터셋 분배 현황 확인 (dataset_stats.py)")
        print(" 10. 데이터셋 split 리밸런싱 (rebalance_dataset_split.py)")
        print("    💡 잘못된 split 비율을 8:1:1로 강제 재조정합니다.")
        print(" 11. AI 모델 ONNX 변환 (onnx_converter.py)")
        print("    💡 PyTorch (.pth) 모델을 웹용 ONNX로 변환합니다.")
        print(" 12. ONNX 모델 양자화 (onnx_quantizer.py)")
        print("    💡 models/onnx 내 미양자화 모델을 INT8로 압축합니다.")
        print(" 13. AI 테스트 분석 리포트 생성 (test_report_generator.py)")
        print("    💡 테스트 결과를 LLM으로 분석하여 리포트를 작성합니다.")
        print("-" * 50)
        print(" 14. 종료 (Exit)")
        print("-" * 50)
        
        choice = input("👉 실행할 작업의 번호를 입력하세요 (1~14): ").strip()

        match choice:
            case "1":
                print("\n🧹 데이터셋 초기화를 시작합니다...")
                run_script("engines.dataset_initializer")
            case "2":
                print("\n🚀 데이터셋 생성을 시작합니다...")
                print("💡 에셋당 생성할 이미지 개수를 입력하세요. (기본값: 300)")
                count_input = input("👉 입력 (엔터 시 300): ").strip()
                
                if count_input == "":
                    run_script("engines.dataset_generator")
                else:
                    run_script("engines.dataset_generator", count_input)
            case "3":
                print("\n🚀 AI 모델 학습을 시작합니다...")
                run_script("engines.model_trainer")
            case "4":
                print("\n🧪 AI 모델 최종 테스트를 시작합니다...")
                run_script("engines.model_tester")
            case "5":
                print("\n🚀 GCP Storage 업로드를 시작합니다...")
                print("💡 기존 파일을 덮어씌우시겠습니까? (y: 덮어쓰기 / n: 건너뛰기)")
                ov_input = input("👉 입력 (기본값 y): ").strip().lower()
                
                if ov_input == "n":
                    run_script("engines.image_uploader", "false")
                else:
                    run_script("engines.image_uploader", "true")
            case "6":
                print("\n🚀 AI 모델 GCS 업로드를 시작합니다...")
                print("💡 기존 파일을 덮어씌우시겠습니까? (y: 덮어쓰기 / n: 건너뛰기)")
                ov_input = input("👉 입력 (기본값 y): ").strip().lower()
                
                if ov_input == "n":
                    run_script("engines.model_uploader", "false")
                else:
                    run_script("engines.model_uploader", "true")
            case "7":
                print("\n🚀 GCP Storage 데이터셋 다운로드를 시작합니다...")
                run_script("engines.dataset_downloader")
            case "8":
                print("\n🚀 GCP Storage 모델 다운로드를 시작합니다...")
                run_script("engines.model_downloader")
            case "9":
                print("\n📊 데이터셋 통계 확인을 시작합니다...")
                run_script("utils.dataset_stats")
            case "10":
                print("\n🔄 데이터셋 리밸런싱을 시작합니다...")
                run_script("utils.rebalance_dataset_split")
            case "11":
                print("\n🔄 AI 모델 ONNX 변환을 시작합니다...")
                run_script("engines.onnx_converter")
            case "12":
                print("\n⚖️ ONNX 모델 양자화를 시작합니다...")
                run_script("utils.onnx_quantizer")
            case "13":
                print("\n📝 AI 테스트 분석 리포트 생성을 시작합니다...")
                run_script("utils.test_report_generator")
            case "14" | "exit" | "quit":
                print("\n👋 프로그램을 종료합니다. 즐거운 하루 되세요!")
                break
            case "":
                continue
            case _:
                print(f"\n❗ '{choice}'은(는) 잘못된 입력입니다. 1~14번 중에서 선택해주세요.")
