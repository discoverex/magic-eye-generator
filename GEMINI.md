#  StereoVision Showdown: AI vs Human

## 1. 프로젝트 개요

> "AI는 인간처럼 착시를 통해 입체감을 느낄 수 있을까?"

본 프로젝트는 매직아이(SIRDS) 이미지를 해석하는 AI 모델을 구축하고, 학습 데이터량에 따른 지능 지수를 단계별 난이도로 구성하여 인간과 대결하는 게이미피케이션 AI 서비스입니다.

단순한 이미지 생성을 넘어, Computer Vision의 입체 인지 능력을 테스트합니다. AI 모델이 매직아이 내부의 미세한 패턴 차이를 분석하여 3D 형상을 복원(Depth Extraction)하고, 이를 분류(Classification)하는 과정을 게임으로 풀어냈습니다.

## 2. 주요 기술

### 2.1. 라이브러리 및 프레임워크
이 프로젝트는 다음과 같은 주요 기술과 라이브러리를 사용합니다.

- **웹 프레임워크**: `FastAPI`, `Uvicorn`
- **AI / 머신러닝**: `PyTorch`, `ONNX`, `Diffusers`, `Transformers`, `Accelerate`, `Langchain`, `Langchain-OpenAI`
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
├───test_results/             # 모델 최종 테스트용 성능 시각화 리포트 (.png)
├───main/                     # 통합 실행기 로직
│   ├───runner.py             # 메뉴 및 실행 제어 (총 14개 시나리오 지원)
│   └───__init__.py
├───models/                   # AI 모델 가중치 및 로컬 캐시
│   ├───players/              # 단계별 학습된 AI 모델 (.pth)
│   └───onnx/                 # 웹 배포용 변환된 모델 (.onnx)
└───src/                      # 소스 코드 루트
    ├───config/               # 애플리케이션 설정 (settings.py)
    ├───consts/               # 상수 정의 (에셋 정의 등)
    ├───engines/              # 핵심 실행 엔진
    │   ├───dataset_initializer.py # 데이터셋 폴더 초기화
    │   ├───dataset_generator.py   # 매직아이 대량 생성 (8:1:1 split 자동 할당)
    │   ├───dataset_downloader.py  # GCS에서 데이터셋 다운로드
    │   ├───model_trainer.py       # GPU 최적화 기반 AI 모델 단계별 학습
    │   ├───model_tester.py        # PyTorch/ONNX 모델 최종 성능 측정 및 시각화 (업데이트)
    │   ├───model_uploader.py      # AI 모델 GCS 업로드 (ONNX 모델 대상, 덮어쓰기 선택 가능)
    │   ├───model_downloader.py    # GCS에서 모델 다운로드
    │   ├───onnx_converter.py      # PyTorch 모델을 ONNX로 변환 및 GCS 업로드
    │   └───image_uploader.py      # GCS 업로드 (전체 데이터셋 대상, 덮어쓰기 여부 선택 기능)
    ├───dtos/                 # 데이터 전송 객체 (MagicEyeDataset 등)
    ├───services/             # 외부 연동 서비스
    │   ├───gcp_storage_service.py # GCP Storage 연동
    │   ├───hf_storage_service.py  # Hugging Face Hub 연동
    │   └───magic_eye_service.py   # 매직아이 생성 핵심 서비스
    └───utils/                # 공통 유틸리티
        ├───dataset_stats.py       # 데이터셋 분배 통계 및 시각화
        ├───rebalance_dataset_split.py # 데이터셋 split 리밸런싱 (8:1:1)
        ├───test_report_generator.py # AI 테스트 분석 리포트 생성
        ├───onnx_quantizer.py      # ONNX 모델 INT8 양자화 및 최적화
        ├───split_helper.py        # 공통 split 결정 로직
        └───stereogram.py          # 매직아이 생성 핵심 알고리즘
```

## 4. 설치 및 실행

### 4.1. 설치

이 프로젝트는 `uv`를 사용하여 패키지를 관리합니다.

1.  **프로젝트 클론 및 의존성 동기화**
    프로젝트를 처음 클론한 후, 다음 명령어를 통해 가상 환경 생성 및 모든 의존성을 한 번에 설치할 수 있습니다.
    ```bash
    uv sync
    ```

2.  **가상 환경 활성화**
    ```bash
    .venv\Scripts\activate  # Windows
    source .venv/bin/activate  # macOS/Linux
    ```

### 4.2. 실행

이 서비스는 통합 실행기(`main.py`)를 통해 모든 주요 기능을 실행할 수 있습니다.

#### 터미널(Terminal)에서 실행
```bash
python main.py
```

#### IDE에서 실행 (VS Code / PyCharm)
- **VS Code**: `main.py`를 열고 **Run Python File** (재생 아이콘) 또는 `F5` 키로 실행합니다.
- **PyCharm**: `main.py` 우클릭 후 **Run 'main'**을 선택합니다.

실행 후 콘솔의 안내에 따라 번호를 입력하여 데이터셋 관리, 모델 학습 및 평가, 유틸리티 실행 등을 수행할 수 있습니다.

### 5. 개발 지침

- **객체지향 설계 (OOP)**
  - Class 기반 구현: `src/engines` 및 `src/services` 계층의 모든 핵심 로직은 클래스 기반으로 구현되어 상태와 기능을 캡슐화합니다.
  - 모듈화: 데이터 로딩(DTO), 외부 서비스(Service), 핵심 엔진(Engine)을 명확히 분리합니다.
- **데이터 무결성 및 밸런싱**
  - **8:1:1 원칙**: 모든 데이터셋은 `train:val:test = 8:1:1` 비율을 엄격히 준수해야 하며, `src/utils/split_helper.py`를 통해 일관되게 관리합니다.
  - 리밸런싱: 생성 과정에서 비율이 깨진 경우 `rebalance_dataset_split.py`를 통해 즉시 교정합니다.
- **GPU 성능 최적화**
  - 학습(`model_trainer.py`) 및 테스트(`model_tester.py`) 시 **AMP(혼합 정밀도)**와 **DataLoader 병렬화**(num_workers, pin_memory)를 필수 적용하여 실행 속도를 극대화합니다.
- **주석 및 코딩 스타일**
  - 주석은 **한국어**로 작성하는 것을 원칙으로 합니다.
  - `Ruff`를 사용하여 코드 스타일을 유지합니다.
  - **분기 처리**: 조건 분기(Case)가 4개 이상인 경우 `if-else` 대신 `match-case` (Python 3.10+) 문을 사용하여 가독성을 높입니다.
- **Gemini 협업 규칙**
  - 제미나이(AI)와의 모든 대화 및 응답은 한국어로 진행합니다.
  - 코드 제안 시 본 문서에 명시된 OOP 구조와 Ruff 스타일을 반영해야 합니다.
- **LLM 활용 및 자원 최적화**
  - **모델 선택**: 프롬프트 생성과 같은 단순 텍스트 작업에는 고스펙 외부 API(GPT-4o mini 등) 대신 경량 로컬 모델인 `Llama 3.2`를 우선 사용합니다.
  - **리소스 효율성**: `Llama 3.1 (8B)` 모델은 약 4.7GB의 용량과 높은 VRAM을 요구하여 Stable Diffusion 및 DPT 모델과 병행 실행 시 GPU 자원 경합을 유발할 수 있습니다. `Llama 3.2`를 통해 최소한의 자원으로 고품질 프롬프트를 생성함으로써 전체 생성 파이프라인의 처리량(Throughput)을 최적화합니다.
