# AI 플레이어 단계별 학습 시스템 (AI Player Training System)

본 문서는 `StereoVision Showdown` 프로젝트에서 AI 플레이어의 지능 지수를 단계별(Level 1~10)로 구현하기 위한 학습 프로세스를 기술합니다. 이 시스템은 매직아이 이미지를 분류하는 능력을 단계적으로 강화하여 인간과의 대결에서 난이도를 조절하는 역할을 합니다.

---

## 1. 기능의 목적
- **단계별 난이도 구현**: 학습 데이터의 양과 학습 시간을 조절하여 레벨 1(초보)부터 레벨 10(전문가)까지의 AI 모델을 구축.
- **연속 학습(Continual Learning)**: 하위 레벨에서 학습된 지식을 상위 레벨이 계승하여 학습 속도와 최종 성능을 극대화.
- **데이터 불균형 해소**: 다양한 매직아이 카테고리가 균등하게 학습되도록 샘플링 전략을 적용하여 분류 정확도 향상.

---

## 2. 전체 로직 흐름 및 절차

학습 파이프라인은 GPU 자원을 효율적으로 사용하기 위해 **혼합 정밀도(AMP)**와 **데이터 병렬화** 기술이 적용되었습니다.

### Step 1: 데이터 로딩 및 샘플링 (Data Loading)
- **모듈**: `src/dtos/magic_eye_dataset.py`, `src/engines/model_trainer.py`
- **내용**: 
    - `WeightedRandomSampler`를 사용하여 데이터셋 내 클래스 불균형을 해결합니다.
    - `data_ratio` 매개변수를 통해 현재 레벨에 필요한 데이터 양(예: Lv.1은 10%, Lv.10은 100%)만 선택적으로 로드합니다.

### Step 2: 모델 초기화 및 가중치 계승 (Model Initialization)
- **모듈**: `src/engines/model_trainer.py` (`_init_model`)
- **내용**: 
    - **Lv.1**: ImageNet으로 사전 학습된 기본 가중치를 사용하여 시작합니다.
    - **Lv.2 ~ 10**: 직전 레벨(Lv.N-1)에서 저장된 모델 가중치를 로드하여 학습을 이어갑니다.
    - **최적화**: PyTorch 2.0+의 `torch.compile`을 사용하여 모델 연산 그래프를 최적화합니다.

### Step 3: 최적화된 학습 실행 (AMP Training)
- **모듈**: `src/engines/model_trainer.py` (`train_level`)
- **내용**: 
    - **AMP(Automatic Mixed Precision)**: FP16 연산을 혼합하여 VRAM 사용량을 줄이고 학습 속도를 2배 이상 향상시킵니다.
    - **Adam Optimizer**: 학습률(Learning Rate)을 레벨에 따라 미세 조정(Lv.1: 1e-4, Lv.2+: 5e-5)하여 안정적인 수렴을 유도합니다.

### Step 4: 검증 및 조기 종료 (Validation & Early Stopping)
- **내용**: 
    - 매 에포크(Epoch)마다 검증 데이터셋(Val Split)으로 손실(Loss)과 정확도(Accuracy)를 측정합니다.
    - `Patience` 기간 동안 검증 손실이 개선되지 않으면 학습을 조기 종료하여 과적합(Overfitting)을 방지합니다.

---

## 3. 사용된 AI 모델 정보

| 구성 요소 | 역할 | 모델명 | 출처 |
| :--- | :--- | :--- | :--- |
| **Backbone** | 이미지 특징 추출 | `ResNet-18` | [PyTorch torchvision](https://pytorch.org/vision/stable/models.html) |
| **Pretrained** | 사전 학습 가중치 | `ResNet18_Weights.DEFAULT` | ImageNet-1K 가중치 |

---

## [부록] 모델 선정 이유

### 1. ResNet-18
- **이유**: 본 프로젝트의 핵심인 '매직아이 분류'는 이미지의 전반적인 형태와 미세한 패턴의 반복을 동시에 파악해야 합니다. ResNet-18은 깊은 신경망이면서도 파라미터 수가 적어 학습 속도가 매우 빠르며, 10단계에 걸친 연속 학습 시나리오를 수행하기에 시간 및 하드웨어 자원 효율성이 가장 뛰어납니다.

### 2. WeightedRandomSampler 적용
- **이유**: 자체 제작된 매직아이 데이터셋은 생성 과정에서 특정 카테고리의 생성 속도가 다를 수 있습니다. 샘플러를 통해 모든 클래스가 동일한 빈도로 학습되도록 강제함으로써, 특정 사물만 잘 맞추는 편향(Bias) 현상을 방지하고 공정한 난이도를 제공합니다.

### 3. AMP(Automatic Mixed Precision)
- **이유**: 10개의 모델을 순차적으로 학습시켜야 하는 프로젝트 특성상 전체 학습 시간 단축이 필수적입니다. AMP를 사용하면 모델의 예측 정확도는 유지하면서 GPU 연산 처리량(Throughput)을 극대화할 수 있어, 전체 개발 주기를 획기적으로 단축할 수 있습니다.
