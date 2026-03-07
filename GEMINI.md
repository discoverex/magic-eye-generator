#  StereoVision Showdown: AI vs Human

## 1. 프로젝트 개요

> "AI는 인간처럼 착시를 통해 입체감을 느낄 수 있을까?"

본 프로젝트는 매직아이(SIRDS) 이미지를 해석하는 AI 모델을 구축하고, 학습 데이터량에 따른 지능 지수를 단계별 난이도로 구성하여 인간과 대결하는 게이미피케이션 AI 서비스입니다.

단순한 이미지 생성을 넘어, Computer Vision의 입체 인지 능력을 테스트합니다. AI 모델이 매직아이 내부의 미세한 패턴 차이를 분석하여 3D 형상을 복원(Depth Extraction)하고, 이를 분류(Classification)하는 과정을 게임으로 풀어냈습니다.

## 2. 주요 기술

### 2.1. 라이브러리 및 프레임워크
이 프로젝트는 다음과 같은 주요 기술과 라이브러리를 사용합니다.

- **웹 프레임워크**: `FastAPI`, `Uvicorn`
- **AI / 머신러닝**: `PyTorch`, `TensorFlow`, `Diffusers`, `Transformers`, `Accelerate`, `Langchain`, `Langchain-OpenAI` (README.md에는 TensorFlow가 언급되나, 현재 `pyproject.toml`에는 PyTorch 기반으로 구성됨)
- **이미지 및 데이터 처리**: `NumPy`, `Pillow`, `OpenCV`
- **클라우드 서비스**: `Firebase Admin`, `Google Cloud Storage`
- **기타 유틸리티**: `python-dotenv` (환경 변수), `googletrans` (번역), `httpx` (HTTP 클라이언트)
- **패키지 및 환경 관리**: `uv`

### 2.2. 데이터셋
- **자체 제작 SIRDS 데이터셋**: Depth Map과 Random Dot Pattern을 결합한 매직아이 자동 생성 파이프라인을 통해 구축된 자체 데이터셋을 사용합니다.

## 3. 프로젝트 구조

```
.
│   .env.example              # 환경 변수 예시 파일
│   .gitignore
│   .python-version           # 파이썬 버전 명시
│   main.py                   # 간단한 실행 스크립트
│   pyproject.toml            # 프로젝트 설정 및 의존성 관리
│   README.md
│   uv.lock                   # uv 잠금 파일
│
└───src/                      # 소스 코드 루트
    │   main.py               # FastAPI 애플리케이션의 메인 진입점
    │
    ├───config/               # 애플리케이션 설정
    │   └───settings.py
    │
    ├───consts/               # 상수 정의
    │   └───magic_eye_assets.py
    │
    ├───core/                 # 핵심 로직
    │   ├───generator.py      # 매직아이 이미지 생성 로직
    │   └───trainer.py        # 난이도 예측 모델 학습 로직
    │
    ├───dtos/                 # 데이터 전송 객체 (Data Transfer Objects)
    │   └───generated_image.py
    │
    ├───services/             # 외부 서비스 연동
    │   ├───gcp_storage.py
    │   └───magic_eye_service.py
    │
    └───utils/                # 보조 유틸리티
        ├───get_diverse_prompts.py
        └───stereogram.py
```

## 4. 설치 및 실행

### 4.1. 설치

이 프로젝트는 `uv`를 사용하여 패키지를 관리합니다.

1.  **가상 환경 생성 및 활성화**
    ```bash
    # 가상 환경 생성
    uv venv

    # 가상 환경 활성화 (Windows)
    .venv\Scripts\activate

    # 가상 환경 활성화 (macOS/Linux)
    source .venv/bin/activate
    ```

2.  **의존성 패키지 설치**
    `pyproject.toml`에 명시된 의존성을 설치합니다. 개발 환경에서는 `dev` 그룹을 포함하여 설치하는 것을 권장합니다.

    ```bash
    # -e 옵션으로 소스 코드를 편집 가능 모드로 설치
    uv pip install -e ".[dev]"
    ```
    > **참고:** Windows 환경에서는 PyTorch가 CUDA 버전에 맞게 자동으로 설치됩니다 (`pyproject.toml` 설정 참고).

### 4.2. 실행

이 서비스는 무거운 AI 모델을 다루는 특성 상 로컬 개발 환경 또는 코랩에서 실행 목적으로 개발되었습니다. 특정 실행 파일을 찾아 실행해주시기 바랍니다.

## 5. 개발 지침

- **객체지향 설계 (OOP)**
  - Class 기반 구현: Router와 Service 계층은 함수형이 아닌 클래스 선언을 원칙으로 합니다.
  - 메서드 구현: 각 기능을 클래스 내부 메서드로 구현하여 응집도를 높이고 상태 관리를 명확히 합니다.
- **주석**
  - 코드의 가독성을 높이기 위해, 주석은 가급적 **한국어**로 작성하는 것을 원칙으로 합니다. 복잡한 로직이나 협업에 필요한 설명은 명확하고 상세하게 기술합니다.
- **코딩 스타일**: 
  - `Ruff`를 사용하여 코드 스타일을 일관되게 유지합니다. 커밋 전에 `ruff check .` 및 `ruff format .` 명령을 실행하는 것을 권장합니다.
  - 3항 연산자 활용: 단순한 조건식은 `true_value if condition else false_value` 형태의 파이썬 3항식을 사용하여 간결함을 유지합니다.
- **코드 품질 관리 (Linting)**
  - Ruff 사용: 프로젝트의 모든 코드는 Ruff를 통해 린팅 및 포맷팅을 수행합니다.
  - Rule 준수: `pyproject.toml`에 선언된 규칙을 엄격히 따르며, 커밋 전 반드시 린트 체크를 권장합니다.
- **환경 변수**
  - 민감한 정보(API 키 등)는 `.env` 파일에 작성하여 관리합니다.
  - `.env.example` 파일을 복사하여 사용하세요.
- **Gemini 협업 및 응답 규칙**
  - 언어 설정: 제미나이(AI)와의 모든 대화 및 제미나이의 모든 응답은 한국어로 진행합니다.
  - 코드 가이드: 제미나이는 코드를 제안할 때 본 문서에 명시된 OOP 구조와 Ruff 스타일을 반영해야 합니다.