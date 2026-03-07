#  StereoVision Showdown: AI vs Human

## 1. 프로젝트 개요

> "AI는 인간처럼 착시를 통해 입체감을 느낄 수 있을까?"

본 프로젝트는 매직아이(SIRDS) 이미지를 해석하는 AI 모델을 구축하고, 학습 데이터량에 따른 지능 지수를 단계별 난이도로 구성하여 인간과 대결하는 게이미피케이션 AI 서비스입니다.

단순한 이미지 생성을 넘어, Computer Vision의 입체 인지 능력을 테스트합니다. AI 모델이 매직아이 내부의 미세한 패턴 차이를 분석하여 3D 형상을 복원(Depth Extraction)하고, 이를 분류(Classification)하는 과정을 게임으로 풀어냈습니다.

## 2. 주요 기술

### 2.1. 라이브러리 및 프레임워크
이 프로젝트는 다음과 같은 주요 기술과 라이브러리를 사용합니다.

- **웹 프레임워크**: `FastAPI`, `Uvicorn`
- **AI / 머신러닝**: `PyTorch`, `Diffusers`, `Transformers`, `Accelerate`, `Langchain`, `Langchain-OpenAI`
- **이미지 및 데이터 처리**: `NumPy`, `Pillow`, `OpenCV`, `Matplotlib` (성능 시각화)
- **클라우드 서비스**: `Firebase Admin`, `Google Cloud Storage`
- **패키지 및 환경 관리**: `uv`

### 2.2. 데이터셋
- **자체 제작 SIRDS 데이터셋**: Depth Map과 Random Dot Pattern을 결합한 매직아이 자동 생성 파이프라인을 통해 구축된 자체 데이터셋을 사용합니다.
- **Data Splitting**: `metadata.csv` 내에 `train`, `val`, `test` 분할 정보를 포함하여 학습, 검증, 최종 성능 측정을 엄격히 분리합니다.

## 3. 프로젝트 구조

```
.
│   main.py                   # 애플리케이션 통합 실행기 (Entry Point)
│   pyproject.toml            # 프로젝트 설정 및 의존성 관리
│   README.md
│   uv.lock                   # uv 잠금 파일
│
├───datasets/                 # 매직아이 데이터셋 저장 경로
├───evaluate_results/         # 모델 검증용 성능 시각화 리포트 (.png)
├───test_results/             # 모델 최종 테스트용 성능 시각화 리포트 (.png)
├───main/                     # 통합 실행기 로직
│   ├───runner.py             # 메뉴 및 실행 제어 (총 7개 시나리오 지원)
│   └───__init__.py
├───models/                   # AI 모델 가중치 및 로컬 캐시
│   └───players/              # 단계별 학습된 AI 모델 (.pth)
└───src/                      # 소스 코드 루트
    ├───config/               # 애플리케이션 설정 (settings.py)
    ├───consts/               # 상수 정의 (에셋 정의 등)
    ├───core/                 # 핵심 비즈니스 로직 (Class 기반 리팩토링 완료)
    │   ├───dataset_initializer.py # 데이터셋 폴더 초기화
    │   ├───dataset_generator.py   # 매직아이 대량 생성 (개수 동적 설정)
    │   ├───trainer.py             # ResNet-18 기반 AI 모델 단계별 학습
    │   ├───model_evaluator.py     # 검증 데이터 기반 모델 성능 평가 및 시각화
    │   ├───model_tester.py        # 테스트 데이터 기반 최종 모델 성능 측정 및 시각화
    │   └───image_uploader.py      # GCS 업로드 (Test 데이터 선별 업로드)
    ├───dtos/                 # 데이터 전송 객체 (MagicEyeDataset 등)
    ├───services/             # 외부 연동 서비스 (GCPStorageService 등)
    └───utils/                # 공통 유틸리티 (Stereogram 생성 로직 등)
```

## 4. 설치 및 실행

### 4.1. 설치

이 프로젝트는 `uv`를 사용하여 패키지를 관리합니다.

1.  **가상 환경 생성 및 활성화**
    ```bash
    uv venv
    .venv\Scripts\activate  # Windows
    source .venv/bin/activate  # macOS/Linux
    ```

2.  **의존성 패키지 설치**
    ```bash
    uv pip install -e ".[dev]"
    ```

### 4.2. 실행

이 서비스는 통합 실행기(`main.py`)를 통해 모든 주요 기능을 실행할 수 있습니다.

```bash
# 통합 메뉴 실행
python main.py
```

실행 후 콘솔의 안내에 따라 1~7번 번호를 입력하여 데이터셋 초기화, 생성, 모델 학습, 평가, 테스트 또는 GCP 업로드를 수행할 수 있습니다.

### 5. 개발 지침

- **객체지향 설계 (OOP)**
  - Class 기반 구현: `src/core` 및 `src/services` 계층의 모든 핵심 로직은 클래스 기반으로 구현되어 상태와 기능을 캡슐화합니다.
  - 모듈화: 데이터 로딩(DTO), 외부 서비스(Service), 핵심 로직(Core)을 명확히 분리합니다.
- **데이터 무결성**
  - AI 훈련에는 `split='train'` 데이터만 사용하며, 서비스에 사용되는 GCS 업로드 데이터는 `split='test'` 데이터만 선별하여 공정성을 보장합니다.
- **주석 및 코딩 스타일**
  - 주석은 **한국어**로 작성하는 것을 원칙으로 합니다.
  - `Ruff`를 사용하여 코드 스타일을 유지합니다.
- **Gemini 협업 규칙**
  - 제미나이(AI)와의 모든 대화 및 응답은 한국어로 진행합니다.
  - 코드 제안 시 본 문서에 명시된 OOP 구조와 Ruff 스타일을 반영해야 합니다.
