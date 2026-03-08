import os
import shutil

from src.config.settings import BASE_DIR


class DatasetInitializer:
    """
    datasets 폴더 내의 모든 데이터(이미지 및 메타데이터)를 삭제하여 초기화하는 클래스
    """
    def __init__(self):
        self.dataset_dir = os.path.join(BASE_DIR, "datasets")

    def run(self):
        """
        사용자 확인 후 datasets 폴더 내의 모든 내용을 삭제합니다.
        """
        if not os.path.exists(self.dataset_dir):
            print(f"📂 '{self.dataset_dir}' 폴더가 존재하지 않습니다. 초기화할 작업이 없습니다.")
            return

        print(f"\n⚠️  데이터셋 초기화 경고")
        print(f"대상 경로: {self.dataset_dir}")
        print("🚨 이 작업은 복구할 수 없습니다. 모든 이미지와 metadata.csv 파일이 영구적으로 삭제됩니다.")
        
        confirm = input("👉 정말로 모든 데이터를 삭제하시겠습니까? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ 초기화 작업이 취소되었습니다.")
            return

        print("🧹 데이터 삭제 중...")
        try:
            # 폴더 내의 모든 항목 순회하며 삭제
            for filename in os.listdir(self.dataset_dir):
                file_path = os.path.join(self.dataset_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path) # 파일 또는 심볼릭 링크 삭제
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path) # 폴더 삭제
                except Exception as e:
                    print(f"  ❌ '{filename}' 삭제 중 오류 발생: {e}")

            print("✨ 데이터셋 폴더가 성공적으로 초기화되었습니다.")
        except Exception as e:
            print(f"❌ 초기화 중 치명적인 오류 발생: {e}")

if __name__ == "__main__":
    initializer = DatasetInitializer()
    initializer.run()
