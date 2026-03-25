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
- AI/ML: PyTorch, ONNX, OpenCV, Diffusers, Transformers
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

## 🦙 Local LLM for Prompt Generation (Ollama)

본 프로젝트는 매직아이 생성을 위한 **Stable Diffusion** 프롬프트를 자동으로 생성하기 위해 로컬 LLM인 **Ollama**를 활용합니다. 

### 1. 왜 로컬 LLM이 필요한가?
- **프롬프트 다양성**: 단순한 키워드 반복이 아닌, LLM을 통해 사물의 배치, 각도, 질감을 묘사하는 풍부한 영문 프롬프트를 생성하여 학습 데이터의 품질을 높입니다.
- **비용 및 프라이버시**: 외부 API(OpenAI 등)를 사용하지 않고 로컬 환경에서 무제한으로 프롬프트를 생성할 수 있습니다.

### 2. 왜 `llama3.2` 모델인가? (Efficiency)
- **자원 최적화**: 기존 `gpt-40-mini`는 단순 텍스트 작업에 과한 고수준 LLM 적용으로 인해 토큰과 리소스 낭비를 유발했습니다. 또한, 같은 ollama 계열 모델이더라도 일반적인 `llama3.1:8b` 모델은 약 4.7GB의 용량과 높은 VRAM을 요구하여 Stable Diffusion 학습/생성과 병행하기에 무거울 수 있습니다.
- **성능 유지**: `llama3.2` (3B/1B) 모델은 훨씬 가벼우면서도 프롬프트 생성과 같은 단순 텍스트 작업에서 충분한 성능을 발휘합니다. 이를 통해 GPU 자원을 이미지 생성 엔진에 더 많이 할당할 수 있습니다.

### 3. 설치 및 설정 가이드

#### Ollama 설치
[Ollama 공식 웹사이트](https://ollama.com/)에서 운영체제에 맞는 설치 파일을 내려받아 설치합니다.

#### 모델 다운로드 (권장: llama3.2)
터미널에서 아래 명령어를 실행하여 모델을 내려받습니다.
```bash
# 기본 3B 모델 (성능과 속도의 균형)
ollama run llama3.2

# 또는 초경량 1B 모델 (가장 빠른 속도)
ollama run llama3.2:1b
```

#### 환경 변수 설정
프로젝트 루트의 `.env` 파일에 사용 중인 모델 이름을 명시합니다.
```env
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

## 🌐 AI Model (ONNX & GCP Storage)

학습된 10단계의 AI 모델은 웹 환경(Next.js, Node.js)에서 직접 실행할 수 있도록 **ONNX** 형식으로 변환되어 **GCP Storage**에 호스팅됩니다. 별도의 파이썬 백엔드 없이 클라이언트 브라우저에서 `onnxruntime-web`을 통해 실시간 추론이 가능합니다.

### 1. 모델 로드 및 추론 (Next.js / TypeScript Example)

프론트엔드에서 특정 레벨(1~10)의 ONNX 모델을 다운로드하여 이미지를 판독하는 예제입니다.

#### 패키지 설치
```bash
npm install onnxruntime-web
```

#### 추론 로직 (TypeScript)
```typescript
import * as ort from 'onnxruntime-web';

/**
 * GCS에서 ONNX 모델을 로드하여 매직아이 이미지를 분석합니다.
 */
export async function predictMagicEye(
  imageTensor: ort.Tensor, // 전처리된 (1, 3, 224, 224) 텐서
  level: number = 10
): Promise<{ label: string; score: number }> {
  // 1. GCS 버킷에서 해당 레벨의 모델 로드
  const modelUrl = `https://storage.googleapis.com/YOUR_BUCKET_NAME/models/onnx/ai_lv${level}.onnx`;
  const session = await ort.InferenceSession.create(modelUrl);

  // 2. 추론 실행
  const inputs = { input: imageTensor };
  const outputs = await session.run(inputs);
  const output = outputs.output; // 모델의 출력 레이어 이름

  // 3. 결과 해석 (Softmax 및 Label 매칭)
  const probabilities = softmax(output.data as Float32Array);
  const maxProb = Math.max(...probabilities);
  const classIdx = probabilities.indexOf(maxProb);
  
  // ASSETS_LABELS는 모델 학습 시 사용된 클래스 순서와 동일해야 함
  const ASSETS_LABELS = [
    "dinosaur", "elephant", "rocket", "guitar", "umbrella", 
    "butterfly", "bicycle", "heart", "star", "tree", 
    "car", "house", "apple", "book", "chair", 
    "cup", "diamond", "fish", "flower", "hat", 
    "key", "leaf", "moon", "mountain", "pencil", 
    "phone", "ship", "sun", "watch", "cloud"
  ];

  const label = ASSETS_LABELS[classIdx];

  return { label, score: maxProb };
}

function softmax(arr: Float32Array): number[] {
  const maxVal = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - maxVal));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return Array.from(exps.map((x) => x / sumExps));
}
```

### 2. 응답 데이터 형태 (Response Format)

추론 함수가 반환하는 최종 객체 구조입니다.

```json
{
  "label": "heart",
  "score": 0.9842
}
```

- **label**: `src/consts/magic_eye_assets.py`에 정의된 30종의 사물 ID (예: `dinosaur`, `rocket`, `apple` 등)
- **score**: 0~1 사이의 확률 값 (신뢰도)

### 3. AI 플레이어 난이도 가이드

변환된 ONNX 모델은 각 레벨별 학습 상태를 그대로 유지하며, GCS에서 필요한 시점에 동적으로 로드됩니다.

| 레벨 | 모델 파일 | 특징 |
|:---:|:---:|---|
| **Lv 1-3** | `ai_lv1~3.onnx` | 노이즈와 형상을 거의 구분하지 못하는 초보 단계입니다. |
| **Lv 4-7** | `ai_lv4~7.onnx` | 대략적인 외곽선은 파악하지만 복잡한 패턴에서 실수를 합니다. |
| **Lv 8-10** | `ai_lv8~10.onnx` | 고해상도 시차 분석을 통해 인간보다 정밀하게 정답을 맞춥니다. |

---

## 프로젝트 구조

```
.
│   main.py                   # 애플리케이션 통합 실행기 (Entry Point)
│   pyproject.toml            # 프로젝트 설정 및 의존성 관리
│   README.md
│   uv.lock                   # uv 잠금 파일
│
├───datasets/                 # 매직아이 데이터셋 저장 경로
├───test_results/             # 모델 최종 테스트 시각화 결과 저장
├───main/                     # 통합 실행기 로직
│   ├───runner.py             # 메뉴 및 실행 제어 로직 (총 10개 옵션 제공)
│   └───__init__.py
├───models/                   # AI 모델 가중치 및 로컬 캐시
│   ├───players/              # 학습된 AI 플레이어 모델 (.pth)
│   └───onnx/                 # 웹용 변환된 모델 (.onnx)
└───src/                      # 소스 코드 루트
    ├───config/               # 애플리케이션 설정
    ├───consts/               # 상수 및 에셋 정의
    ├───engines/              # 핵심 엔진
    │   ├───dataset_initializer.py # 데이터셋 초기화
    │   ├───dataset_generator.py   # 매직아이 생성 (8:1:1 자동 할당)
    │   ├───dataset_downloader.py  # GCS에서 데이터셋 다운로드
    │   ├───model_trainer.py       # AI 모델 학습
    │   ├───model_tester.py        # PyTorch/ONNX 모델 성능 측정
    │   ├───model_downloader.py    # GCS에서 AI 모델 다운로드
    │   ├───onnx_converter.py      # 모델 ONNX 변환 및 GCS 업로드
    │   ├───model_uploader.py      # AI 모델 GCS 업로드 (ONNX 모델 대상)
    │   └───image_uploader.py      # GCS 업로드 (전체 데이터 업로드)
    ├───dtos/                 # 데이터 전송 객체
    ├───services/             # 외부 서비스 연동
    └───utils/                # 보조 유틸리티
```

## 설치 및 실행

### 설치

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

### 실행

이 서비스는 통합 실행기(`main.py`)를 통해 모든 주요 기능을 실행할 수 있습니다.

#### 터미널(Terminal)에서 실행
```bash
python main.py
```

#### IDE에서 실행

**VS Code**
1. `main.py` 파일을 엽니다.
2. 오른쪽 상단의 **Run Python File** 버튼(재생 아이콘)을 클릭하거나 `F5` 키를 눌러 실행합니다.
3. 하단 터미널(Terminal) 창에서 안내에 따라 번호를 입력하여 기능을 선택합니다.

**PyCharm**
1. `main.py` 파일에서 마우스 오른쪽 버튼을 클릭합니다.
2. **Run 'main'** 메뉴를 선택합니다.
3. 하단 **Run** 탭의 콘솔 창에서 안내에 따라 번호를 입력하여 기능을 선택합니다.

---

**제공 기능:**
1. **데이터셋 초기화**: 기존 생성된 모든 데이터를 삭제합니다.
2. **데이터셋 생성**: 에셋별 생성 개수를 지정하여 매직아이를 대량 생성합니다. (8:1:1 자동 분배)
3. **AI 모델 학습**: 10단계 AI 모델을 학습시킵니다.
4. **AI 모델 최종 테스트**: 테스트 데이터를 통해 모델 성능을 측정합니다. (PyTorch/ONNX 선택 가능)
5. **GCP 업로드**: 로컬 데이터셋 전체를 GCS에 백업 업로드합니다. (기존 파일 덮어쓰기/건너뛰기 선택 가능)
6. **AI 모델 업로드**: 학습된 ONNX 모델들을 GCS의 `models/onnx/` 경로로 업로드합니다. (덮어쓰기 선택 가능)
7. **데이터셋 다운로드**: GCS에 백업된 데이터셋을 로컬로 내려받습니다.
8. **모델 다운로드**: GCS에 저장된 훈련 모델(pth, onnx)을 로컬로 내려받습니다.
9. **데이터셋 통계**: 데이터셋 분배 현황을 시각화합니다.
10. **데이터셋 리밸런싱**: 데이터셋 분배 비율을 8:1:1로 교정합니다.
11. **AI 모델 ONNX 변환**: PyTorch 모델을 웹용 ONNX로 변환하고 GCS에 업로드합니다.
12. **ONNX 모델 양자화**: models/onnx 내 미양자화 모델을 INT8로 압축하여 최적화합니다.
13. **AI 테스트 분석 리포트 생성**: 테스트 결과를 LLM으로 분석하여 상세 리포트를 작성합니다.
14. **종료**: 프로그램을 종료합니다.

