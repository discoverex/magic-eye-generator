# 🧪 StereoVision Showdown: AI 모델 성능 및 프로젝트 목표 부합성 분석 리포트 ({model_type_upper})
**리포트 생성 일시**: {report_date}
**테스트 대상**: AI Player Level 1 ~ 10 ({model_type_upper} 기반)
**테스트 장치**: {device}

![Performance Summary]({img_filename})

## 1. 모델 성능 데이터 요약
| 레벨 | 정확도 (%) | 총 이미지 | 목표 지능 (README 기준) | 평가 |
|:---:|:---:|:---:|:---|:---:|
{summary_table}

---
## 2. 프로젝트 목표 기반 핵심 평가 (Critical Review)

본 프로젝트의 핵심 목적은 **"데이터량에 따른 단계별 지능 지수 구현"**과 **"이를 통한 게이미피케이션(난이도 선택)"**입니다.

### {discrimination_icon} 레벨 간 변별력 분석
*   **최저-최고 레벨 격차**: {gap}%p (Lv.1 {min_acc}% vs Max {max_acc}%)
*   **분석 결과**: {discrimination_result}

{stagnation_section}

### 🔍 에셋별 인지 능력 차이
*   **취약 에셋**: {weak_assets_str}
*   **원인 추정**: 복잡한 외곽선이나 미세한 뎁스 차이를 가진 사물에서 AI의 인지 실패가 집중되고 있습니다.

---
## 3. 종합 진단 및 개선 과제

**현재 모델 상태: "{diagnosis}"**

{diagnosis_detail}

### 🛠 향후 액션 플랜
1. **초기 레벨 지능 강제 하향**: Lv.1~3 모델은 학습 에포크를 1회로 제한하거나 해상도를 64x64로 낮추어 학습시키세요.
2. **데이터 격차 확대**: 레벨 간 학습 데이터 개수를 더 극단적으로 차이나게 배치(예: 10장 vs 1000장) 하세요.
3. **에셋 난이도별 최적화**: 인지하기 어려운 복잡한 에셋을 고레벨 모델에서만 완벽히 학습하도록 조정하세요.

---
*본 보고서는 TestReportGenerator에 의해 데이터 분석 기반으로 자동 생성되었습니다.*
