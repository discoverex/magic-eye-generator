# 🏗️ StereoVision Showdown: AI vs Human Magic Eye Quiz
> "AI는 인간처럼 착시를 통해 입체감을 느낄 수 있을까?" > 본 프로젝트는 매직아이(SIRDS) 이미지를 해석하는 AI 모델을 구축하고, 학습 데이터량에 따른 지능 지수를 단계별 난이도로 구성하여 인간과 대결하는 게이미피케이션 AI 서비스입니다.

---

## 🎯 Project Overview

단순한 이미지 생성을 넘어, Computer Vision의 입체 인지 능력을 테스트합니다. AI 모델이 매직아이 내부의 미세한 패턴 차이를 분석하여 3D 형상을 복원(Depth Extraction)하고, 이를 분류(Classification)하는 과정을 게임으로 풀어냈습니다.

## 🔥 Key Challenges & Solutions

1. 문제 제기 (The Problem)
- 기존 매직아이 생성 알고리즘은 완성도가 너무 높아 기술적 고도화의 여지가 적음.
- 단순히 이미지만 보여주는 서비스는 사용자 인터랙션과 기술적 소구점이 부족함.

2. 해결책 (The Solution)
- AI의 시각 인지 학습: 매직아이를 해석하는 딥러닝 모델을 직접 설계.
- 단계별 파인튜닝: 학습 데이터셋의 규모와 반복 횟수(Epoch)를 조절하여 10단계의 지능형 모델을 구축.
- 대결 시스템: 플레이어가 자신의 실력에 맞는 AI 난이도를 선택해 고득점에 도전하는 퀴즈 게임 구현.
---

## 🎮 Game System

Scoring Logic
- 기본 정답: $100 \times \text{Difficulty Level}$
- 보너스 점수: AI가 오답을 내고 플레이어만 정답을 맞출 경우 $50 \times \text{Difficulty Level}$ 추가 가산점 부여.

AI Difficulty (10 Levels)
- Level 1-3 (Beginner): 노이즈와 형상을 거의 구분하지 못하는 초기 모델.
- Level 4-7 (Intermediate): 대략적인 외곽선은 파악하지만, 복잡한 사물에서 오답 발생.
- Level 8-10 (Expert): 고해상도 시차 분석을 통해 인간보다 빠르게 정답을 맞추는 최적화 모델.

---
## 🛠️ Tech StackFrontend
- React / Next.js (Tailwind CSS)
- Backend: FastAPI (Model Serving)
- AI/ML: PyTorch / TensorFlow, OpenCV
- Dataset: 자체 제작한 SIRDS(Single Image Random Dot Stereograms) 데이터셋

---
## 📦 Model & Asset Management

- Local Weight Storage: 외부 API 의존성을 줄이고 실행 환경의 일관성을 보장하기 위해 모든 사전 학습 모델(Stable Diffusion, DPT)을 ./models 디렉토리에 로컬 캐싱하여 관리함.
- Environment Isolation: 네트워크 단절 상황에서도 모델 서빙이 가능하도록 인프라를 설계함.

---
## 🚀 Technical Points (Showcasing)
- Data Engineering: Depth Map과 Random Dot Pattern을 결합한 매직아이 자동 생성 파이프라인 구축.
- Model Versioning: 파인튜닝 단계별 가중치(Weights) 관리 및 API 서빙 최적화.
- UX/UI: 게임화 요소를 도입하여 AI 모델의 성능 차이를 사용자에게 직관적으로 전달.

---

이 프로젝트가 마음에 드신다면 ⭐️를 눌러주세요!

---

## 프로젝트 구조

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

## 설치 및 실행

### 설치

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

### 실행

이 서비스는 무거운 AI 모델을 다루는 특성 상 로컬 개발 환경 또는 코랩에서 실행 목적으로 개발되었습니다.
특정 실행 파일을 찾아 실행해주시기 바랍니다.