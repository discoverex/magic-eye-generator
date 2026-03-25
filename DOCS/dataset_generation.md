# 매직아이 데이터셋 생성 시스템 (Dataset Generation System)

본 문서는 `StereoVision Showdown` 프로젝트의 핵심 기능인 매직아이(SIRDS) 데이터셋 생성 프로세스를 상세히 기술합니다. 이 시스템은 고품질의 3D 입체 인지 데이터를 대량으로 자동 생성하여 AI 모델 학습에 활용할 수 있도록 설계되었습니다.

---

## 1. 기능의 목적
- **데이터 자동화**: 수동 제작이 어려운 매직아이 이미지를 AI 모델을 활용해 대량으로 생성.
- **다양성 확보**: LLM(Llama 3.2)을 활용해 객체의 각도, 배치, 상태를 다양화한 프롬프트를 생성하여 학습 데이터의 변별력 강화.
- **학습 데이터 구축**: 생성된 매직아이 이미지(Problem)와 대응되는 깊이 맵(Answer)을 쌍으로 구성하여 지도 학습(Supervised Learning) 환경 제공.

---

## 2. 전체 로직 흐름 및 절차

데이터셋 생성은 **GPU 추론(이미지/깊이 생성)**과 **CPU 연산(매직아이 합성/I/O)**의 병목 현상을 최소화하기 위해 비동기 큐와 멀티프로세싱을 결합한 파이프라인으로 동작합니다.

### Step 1: 프롬프트 프리페칭 (Prompt Prefetching)
- **모듈**: `src/utils/get_diverse_prompts.py` (`PromptAgent`)
- **내용**: 백그라운드 워커가 대상 단어(예: 사과, 고래)에 대해 다양한 3D 장면 묘사 프롬프트를 미리 생성하여 큐(Queue)에 쌓아둡니다.
- **모델**: **Llama 3.2** (via Ollama)

### Step 2: 원본 이미지 및 깊이 맵 생성 (GPU Inference)
- **모듈**: `src/services/magic_eye_service.py` (`MagicEyeService`)
- **내용**: 
    1. 생성된 프롬프트를 바탕으로 고대비(High Contrast) 3D 실루엣 이미지를 생성합니다.
    2. 생성된 이미지에서 픽셀 단위의 깊이 정보를 추출하여 깊이 맵(Depth Map)을 생성합니다.
- **모델**: **DreamShaper 8** (이미지 생성), **DPT-Large** (깊이 추출)

### Step 3: 매직아이 합성 (Stereogram Composition)
- **모듈**: `src/utils/stereogram.py` (`create_stereogram`)
- **내용**: 
    - 깊이 맵의 값을 기준으로 랜덤 패턴의 가로 오프셋(Shift)을 계산합니다.
    - NumPy 벡터 연산을 통해 패턴을 반복적으로 배치하여 입체감이 숨겨진 매직아이 이미지를 합성합니다.
- **처리**: CPU 멀티프로세싱 (`_cpu_worker_process`)

### Step 4: 데이터 분할 및 저장 (Splitting & Storage)
- **모듈**: `src/utils/split_helper.py`
- **내용**: 
    - 8:1:1 원칙(`train: 80%`, `val: 10%`, `test: 10%`)에 따라 데이터를 자동으로 분류합니다.
    - 매직아이 이미지와 정답 깊이 맵을 저장하고, `metadata.csv`에 모든 정보를 기록합니다.

---

## 3. 사용된 AI 모델 정보

| 단계 | 역할 | 모델명 | 출처 (HuggingFace 등) |
| :--- | :--- | :--- | :--- |
| **Prompt** | 프롬프트 생성 | `Llama 3.2 (3B)` | [Meta](https://huggingface.co/meta-llama/Llama-3.2-3B) |
| **Image** | 이미지 생성 | `DreamShaper 8` | [Lykon/dreamshaper-8](https://huggingface.co/Lykon/dreamshaper-8) |
| **Depth** | 깊이 맵 추출 | `DPT-Large` | [Intel/dpt-large](https://huggingface.co/Intel/dpt-large) |

---

## [부록] 모델 선정 이유

### 1. Llama 3.2 (3B)
- **이유**: 로컬 환경(Ollama)에서 실행 가능한 경량 모델 중 가장 뛰어난 추론 능력을 보유하고 있습니다. 외부 API 비용 없이 대량의 프롬프트를 빠르게 생성할 수 있으며, 특히 복잡한 레이아웃 지시 사항을 정확히 이해하여 다양한 각도의 3D 프롬프트를 출력하는 데 최적입니다.

### 2. DreamShaper 8
- **이유**: Stable Diffusion v1.5를 기반으로 튜닝된 모델로, 실사보다는 명확한 형태와 질감을 가진 사물 생성에 강점이 있습니다. 매직아이 생성에는 그림자나 복잡한 배경보다 **명확한 실루엣**이 중요한데, DreamShaper는 고대비의 3D 렌더링 스타일을 안정적으로 생성해냅니다.

### 3. DPT-Large (Dense Prediction Transformer)
- **이유**: 단일 이미지에서 깊이를 추정하는 SOTA(State-of-the-Art) 모델 중 하나입니다. 매직아이의 입체감은 깊이 맵의 정교함에 비례하는데, DPT-Large는 객체의 경계선과 미세한 거리 차이를 매우 선명하게 추출하여 최종 매직아이 이미지의 해상도와 입체감을 극대화합니다.
