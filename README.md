# 🏗️ StereoVision Showdown: AI vs Human Magic Eye Quiz
> "AI는 인간처럼 착시를 통해 입체감을 느낄 수 있을까?"

본 프로젝트는 매직아이(SIRDS) 이미지를 해석하는 AI 모델을 구축하고, 학습 데이터량에 따른 지능 지수를 단계별 난이도로 구성하여 인간과 대결하는 게이미피케이션 AI 서비스입니다.

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
- AI/ML: PyTorch, OpenCV, Diffusers, Transformers
- Dataset: 자체 제작한 SIRDS(Single Image Random Dot Stereograms) 데이터셋

---
## 📦 Model & Asset Management

- Local Weight Storage: 외부 API 의존성을 줄이고 실행 환경의 일관성을 보장하기 위해 모든 사전 학습 모델(Stable Diffusion, DPT)을 ./models 디렉토리에 로컬 캐싱하여 관리함.
- Environment Isolation: 네트워크 단절 상황에서도 모델 서빙이 가능하도록 인프라를 설계함.

---
## 🚀 Technical Points (Showcasing)
- Data Engineering: Depth Map과 Random Dot Pattern을 결합한 매직아이 자동 생성 파이프라인 구축.
- Model Versioning: 파인튜닝 단계별 가중치(Weights) 관리 및 API 서빙 최적화.
- Performance Visualization: 모델별 검증(Val) 및 테스트(Test) 결과를 그래프로 시각화하여 성능 추이 관리.

---

## 프로젝트 구조

```
.
│   .env.example              # 환경 변수 예시 파일
│   .gitignore
│   .python-version           # 파이썬 버전 명시
│   main.py                   # 애플리케이션 통합 실행기 (Entry Point)
│   pyproject.toml            # 프로젝트 설정 및 의존성 관리
│   README.md
│   uv.lock                   # uv 잠금 파일
│
├───datasets/                 # 매직아이 데이터셋 저장 경로 (이미지 및 metadata.csv)
├───evaluate_results/         # 모델 검증 성능 시각화 결과 저장
├───test_results/             # 모델 최종 테스트 시각화 결과 저장
├───main/                     # 통합 실행기 로직
│   ├───runner.py             # 메뉴 및 실행 제어 로직 (총 7개 옵션 제공)
│   └───__init__.py
├───models/                   # AI 모델 가중치 및 로컬 캐시
│   ├───players/              # 학습된 AI 플레이어 모델 (.pth)
│   ├───dpt-large/            # Depth Estimation 모델
│   └───stable-diffusion-v1-5 # 이미지 생성 모델
└───src/                      # 소스 코드 루트
    ├───config/               # 애플리케이션 설정
    ├───consts/               # 상수 및 에셋 정의
    ├───core/                 # 핵심 로직 (Class 기반 리팩토링 완료)
    │   ├───dataset_initializer.py # 데이터셋 초기화
    │   ├───dataset_generator.py   # 매직아이 생성 (동적 개수 설정 가능)
    │   ├───trainer.py             # AI 모델 단계별 학습
    │   ├───model_evaluator.py     # 모델 검증 및 시각화
    │   ├───model_tester.py        # 모델 최종 테스트 및 시각화
    │   └───image_uploader.py      # GCS 업로드 (Test 데이터 선별 업로드)
    ├───dtos/                 # 데이터 전송 객체 (MagicEyeDataset 등)
    ├───services/             # 외부 서비스 연동 (GCPStorageService 등)
    └───utils/                # 보조 유틸리티 (Stereogram 생성 로직 등)
```

## 설치 및 실행

### 설치

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

### 실행

이 서비스는 통합 실행기(`main.py`)를 통해 모든 주요 기능을 실행할 수 있습니다.

```bash
python main.py
```

**제공 기능:**
1. **데이터셋 초기화**: 기존 생성된 모든 데이터를 삭제합니다.
2. **데이터셋 생성**: 에셋별 생성 개수를 지정하여 매직아이를 대량 생성합니다.
3. **AI 모델 학습**: 생성된 데이터를 비율별로 사용하여 10단계 AI를 학습시킵니다.
4. **AI 모델 평가**: 검증(Val) 데이터를 통해 학습된 모델의 성능을 시각화합니다.
5. **AI 모델 최종 테스트**: 테스트(Test) 데이터를 통해 모델의 최종 성능을 측정합니다.
6. **GCP 업로드**: 서비스에 사용될 테스트 데이터를 GCS에 업로드합니다.
