# 매직아이 이미지 생성기 (Magic Eye Generator)

## 1. 프로젝트 개요

이 프로젝트는 텍스트 프롬프트를 기반으로 3D 매직아이(스테레오그램) 이미지를 생성하는 것을 목표로 합니다. 생성된 이미지의 게임 난이도를 예측하기 위한 모델 학습 기능도 포함하고 있으며, 적응형 CUDA 지원을 통해 효율적인 GPU 활용을 지향합니다.

## 2. 주요 기술

이 프로젝트는 다음과 같은 주요 기술과 라이브러리를 사용합니다.

- **웹 프레임워크**: `FastAPI`, `Uvicorn`
- **AI / 머신러닝**: `PyTorch`, `Diffusers`, `Transformers`, `Accelerate`, `Langchain`, `Langchain-OpenAI`
- **이미지 및 데이터 처리**: `NumPy`, `Pillow`, `OpenCV`
- **클라우드 서비스**: `Firebase Admin`, `Google Cloud Storage`
- **기타 유틸리티**: `python-dotenv` (환경 변수), `googletrans` (번역), `httpx` (HTTP 클라이언트)
- **패키지 및 환경 관리**: `uv`

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
├───config/                   # 애플리케이션 설정
│   └───settings.py
│
└───src/                      # 소스 코드 루트
    │   main.py               # FastAPI 애플리케이션의 메인 진입점
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

FastAPI 웹 서버를 실행하여 애플리케이션을 구동합니다.

```bash
uvicorn src.main:app --reload
```

> **주의:** 위 명령어를 실행하기 전에 `src/main.py` 파일 내에 `app`이라는 이름의 FastAPI 인스턴스가 생성되어 있어야 합니다.

## 5. 개발 지침

- **객체지향 설계 (OOP)**
  - Class 기반 구현: Router와 Service 계층은 함수형이 아닌 클래스 선언을 원칙으로 합니다.
  - 메서드 구현: 각 기능을 클래스 내부 메서드로 구현하여 응집도를 높이고 상태 관리를 명확히 합니다.
- **주석**
  - 코드의 가독성을 높이기 위해, 주석은 가급적 **한국어**로 작성하는 것을 원칙으로 합니다. 복잡한 로직이나 협업에 필요한 설명은 명확하고 상세하게 기술합니다.
- **코딩 스타일**: 
  - `Ruff`를 사용하여 코드 스타일을 일관되게 유지합니다. 커밋 전에 `ruff check .` 및 `ruff format .` 명령을 실행하는 것을 권장합니다.
  - 3항 연산자 활용: 단순한 조건식은 true_value if condition else false_value 형태의 파이썬 3항식을 사용하여 간결함을 유지합니다.
  - 주석 작성: 코드를 제외한 모든 설명 주석은 한국어로 작성하여 팀 내 의사소통 효율을 높입니다.
- **코드 품질 관리 (Linting)**
  - Ruff 사용: 프로젝트의 모든 코드는 Ruff를 통해 린팅 및 포맷팅을 수행합니다.
  - Rule 준수: pyproject.toml에 선언된 규칙을 엄격히 따르며, 커밋 전 반드시 린트 체크를 권장합니다.
- **환경 변수**
  - 민감한 정보(API 키 등)는 `.env` 파일에 작성하여 관리합니다.
  - `.env.example` 파일을 복사하여 사용하세요.
- **Gemini 협업 및 응답 규칙**
  - 언어 설정: 제미나이(AI)와의 모든 대화 및 제미나이의 모든 응답은 한국어로 진행합니다.
  - 코드 가이드: 제미나이는 코드를 제안할 때 본 문서에 명시된 OOP 구조와 Ruff 스타일을 반영해야 합니다.