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
- Data Engineering: Depth Map과 Random Dot Pattern을 결합한 매직아이 자동 생성 파이프라인 구축 및 **8:1:1 자동 분배 시스템** 구현.
- Model Versioning: 파인튜닝 단계별 가중치(Weights) 관리 및 API 서빙 최적화.
- Performance Optimization: **AMP(혼합 정밀도)** 학습 및 **GPU 병렬 데이터 로딩**을 통해 훈련 및 테스트 속도를 대폭 향상.
- Visualization: 데이터셋 분배 현황 및 모델 성능 추이를 그래프로 시각화하여 데이터 무결성 상시 점검.

## 🌐 AI Model API (Hugging Face)

학습된 10단계의 AI 모델은 Hugging Face Hub에 통합되어 업로드되어 있습니다. `handler.py`(Custom Handler)를 통해 단일 엔드포인트에서 사용자가 원하는 AI 레벨을 동적으로 선택하여 호출할 수 있습니다.

- **Repository**: [postelian/magic-eye-finder](https://huggingface.co/postelian/magic-eye-finder)
- **Endpoint**: `https://api-inference.huggingface.co/models/postelian/magic-eye-finder`

### 1. API 호출 방식 (Next.js / TypeScript Example)

프론트엔드에서 특정 레벨(1~10)의 AI를 선택하여 이미지를 판독하는 예제입니다. 타입 안정성을 위해 `interface`를 정의하여 사용합니다.

```typescript
// types/magic-eye.ts
export interface MagicEyeResponse {
  label: string;
  score: number;
  level: number;
}

// services/magic-eye-api.ts
export async function queryMagicEye(
  imageFile: File | Blob, 
  aiLevel: number = 10
): Promise<MagicEyeResponse[]> {
  const HF_TOKEN = process.env.NEXT_PUBLIC_HF_TOKEN;
  const REPO_ID = "postelian/magic-eye-finder";

  // 이미지를 ArrayBuffer로 변환
  const arrayBuffer = await imageFile.arrayBuffer();

  const response = await fetch(
    `https://api-inference.huggingface.co/models/${REPO_ID}`,
    {
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
      },
      method: "POST",
      body: JSON.stringify({
        inputs: Array.from(new Uint8Array(arrayBuffer)), // 이미지 데이터 전달
        parameters: { level: aiLevel }, // 1~10 사이의 난이도 선택
      }),
    }
  );

  if (!response.ok) {
    throw new Error("AI 분석 요청에 실패했습니다.");
  }

  const result: MagicEyeResponse[] = await response.json();
  return result;
}
```

### 2. 응답 데이터 형태 (Response Format)

커스텀 핸들러를 통해 분석 결과와 함께 요청한 AI 레벨 정보가 포함되어 반환됩니다.

```json
[
  {
    "label": "heart",
    "score": 0.9842,
    "level": 10
  }
]
```

- **label**: `src/consts/magic_eye_assets.py`에 정의된 30종의 사물 ID (예: `dinosaur`, `rocket`, `apple` 등)
- **score**: 0~1 사이의 확률 값 (신뢰도)
- **level**: 추론에 사용된 AI 플레이어의 난이도 레벨 (1~10)

### 3. AI 플레이어 난이도 가이드

사용자가 선택한 레벨에 따라 Hugging Face 서버 내부에서 해당 모델 가중치를 즉시 로드하여 처리합니다.

| 레벨 | 학습 단계 | 내부 동작 및 특징 |
|:---:|:---:|---|
| **Lv 1-3** | 초기 학습 | `ai_lv1~3.pth` 로드: 노이즈와 형상을 거의 구분하지 못하는 초보 단계입니다. |
| **Lv 4-7** | 중간 학습 | `ai_lv4~7.pth` 로드: 대략적인 외곽선은 파악하지만 복잡한 패턴에서 실수를 합니다. |
| **Lv 8-10** | 최종 최적화 | `ai_lv8~10.pth` 로드: 고해상도 시차 분석을 통해 인간보다 정밀하게 정답을 맞춥니다. |

---

## 프로젝트 구조

```
.
│   main.py                   # 애플리케이션 통합 실행기 (Entry Point)
│   pyproject.toml            # 프로젝트 설정 및 의존성 관리
│   README.md
│   uv.lock                   # uv 잠금 파일
│
├───datasets/                 # 매직아이 데이터셋 저장 경로 (이미지 및 metadata.csv)
├───evaluate_results/         # 모델 검증 성능 시각화 결과 저장
├───test_results/             # 모델 최종 테스트 시각화 결과 저장
├───main/                     # 통합 실행기 로직
│   ├───runner.py             # 메뉴 및 실행 제어 로직 (총 9개 옵션 제공)
│   └───__init__.py
├───models/                   # AI 모델 가중치 및 로컬 캐시
│   ├───players/              # 학습된 AI 플레이어 모델 (.pth)
└───src/                      # 소스 코드 루트
    ├───config/               # 애플리케이션 설정
    ├───consts/               # 상수 및 에셋 정의
    ├───engines/              # 핵심 엔진
    │   ├───dataset_initializer.py # 데이터셋 초기화
    │   ├───dataset_generator.py   # 매직아이 생성 (8:1:1 자동 할당)
    │   ├───model_trainer.py       # GPU 가속 기반 AI 모델 학습
    │   ├───model_tester.py        # GPU 가속 기반 모델 성능 측정
    │   ├───model_uploader.py      # AI 모델 Hugging Face 업로드
    │   └───image_uploader.py      # GCS 업로드
    ├───dtos/                 # 데이터 전송 객체
    ├───services/             # 외부 서비스 연동
    └───utils/                # 보조 유틸리티
        ├───dataset_stats.py       # 데이터셋 통계 및 시각화
        ├───rebalance_dataset_split.py # 데이터셋 리밸런싱
        └───split_helper.py        # 공통 split 결정 로직
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
2. **데이터셋 생성**: 에셋별 생성 개수를 지정하여 매직아이를 대량 생성합니다. (8:1:1 자동 분배)
3. **AI 모델 학습**: GPU 가속 및 AMP를 적용하여 10단계 AI를 효율적으로 학습시킵니다.
4. **AI 모델 최종 테스트**: 테스트(Test) 데이터를 통해 모델의 최종 성능을 측정합니다.
5. **GCP 업로드**: 서비스에 사용될 테스트 데이터를 GCS에 업로드합니다.
6. **AI 모델 업로드**: AI 플레이어 모델을 Hugging Face에 업로드합니다.
7. **데이터셋 통계**: 데이터셋의 전체 및 에셋별 분배 현황을 시각화합니다.
8. **데이터셋 리밸런싱**: 잘못된 데이터셋 분배 비율을 8:1:1로 즉시 교정합니다.
9. **종료**: 프로그램을 안전하게 종료합니다.
