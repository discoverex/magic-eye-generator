import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config.settings import BASE_DIR, OPENAI_API_KEY


class TestReportGenerator:
    """
    테스트 결과 데이터를 분석하여 프로젝트 목표(게이미피케이션) 관점의 상세 마크다운 리포트를 생성하는 유틸리티.
    LLM(OpenAI)을 지원하며, LLM이 없는 경우 규칙 기반 분석(Rule-based Analysis)을 수행합니다.
    """
    def __init__(self, run_dir: Optional[os.PathLike] = None):
        self.base_result_dir = BASE_DIR / "test_results"
        self.run_dir = run_dir if run_dir else self.base_result_dir
        
        # LLM 설정 (OpenAI)
        try:
            from langchain_openai import ChatOpenAI
            if OPENAI_API_KEY:
                self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.5)
            else:
                self.llm = None
        except ImportError:
            self.llm = None

    def generate_from_data(self, data: Dict[str, Any], timestamp: str):
        """
        메모리 상의 테스트 데이터를 직접 받아 리포트를 생성합니다. (ModelTester 연동용)
        """
        model_type = data.get("model_type", "pth").upper()
        filename = f"test_report_{model_type.lower()}_{timestamp}.md"
        save_path = os.path.join(self.run_dir, filename)

        if self.llm:
            content = self._generate_llm_report(data)
        else:
            content = self._generate_rule_based_report(data)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"📝 테스트 분석 보고서 생성 완료: {save_path}")
        return save_path

    def _analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터의 수치적 특성을 분석하여 인사이트를 도출합니다."""
        results = data["results"]
        accuracies = [r['accuracy'] for r in results]
        levels = [r['level'] for r in results]
        
        min_acc, max_acc = min(accuracies), max(accuracies)
        gap = max_acc - min_acc
        
        # 성능 역전/정체 구간
        stagnation = []
        for i in range(1, len(accuracies)):
            if accuracies[i] <= accuracies[i-1]:
                stagnation.append(levels[i])
        
        # 에셋 분석 (마지막 레벨 기준)
        last_res = results[-1]
        asset_metrics = last_res.get("per_asset_metrics", {})
        sorted_assets = sorted(asset_metrics.items(), key=lambda x: x[1]["accuracy"])
        
        return {
            "gap": round(gap, 2),
            "min_acc": min_acc,
            "max_acc": max_acc,
            "stagnation": stagnation,
            "weak_assets": sorted_assets[:3],
            "strong_assets": sorted_assets[-3:],
            "is_effective": gap > 10 and not stagnation
        }

    def _generate_rule_based_report(self, data: Dict[str, Any]) -> str:
        """규칙 기반의 비판적 분석 리포트를 생성합니다."""
        analysis = self._analyze_data(data)
        m_type = data["model_type"].lower()
        timestamp = data.get("timestamp", "unknown")
        
        report = []
        report.append(f"# 🧪 StereoVision Showdown: AI 모델 성능 및 프로젝트 목표 부합성 분석 리포트 ({m_type.upper()})")
        report.append(f"**리포트 생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**테스트 대상**: AI Player Level 1 ~ 10 ({m_type.upper()} 기반)")
        report.append(f"**테스트 장치**: {data.get('device', 'unknown')}")

        report.append("\n## 1. 모델 성능 데이터 요약")
        # 이미지 파일명: test_accuracy_summary_{model_type}_{timestamp}.png
        img_filename = f"test_accuracy_summary_{m_type}_{timestamp}.png"
        report.append(f"![Performance Summary]({img_filename})")
        report.append("\n| 레벨 | 정확도 (%) | 총 이미지 | 목표 지능 (README 기준) | 평가 |")
        report.append("|:---:|:---:|:---:|:---|:---:|")

        for r in data["results"]:
            lv = r['level']
            target = "Beginner" if lv <= 3 else "Intermediate" if lv <= 7 else "Expert"
            status = "🟢 적합" if r['accuracy'] > (lv * 5 + 50) else "🟡 미흡"
            report.append(f"| **Lv.{lv}** | {r['accuracy']}% | {r['total_images']} | {target} 모델 | {status} |")

        report.append("\n---\n## 2. 프로젝트 목표 기반 핵심 평가 (Critical Review)")
        report.append("\n본 프로젝트의 핵심 목적은 **\"데이터량에 따른 단계별 지능 지수 구현\"**과 **\"이를 통한 게이미피케이션(난이도 선택)\"**입니다.")

        # 변별력 분석
        gap = analysis['gap']
        report.append(f"\n### {'✅' if gap > 15 else '⚠️ [주의]' if gap > 5 else '❌ [심각]'} 레벨 간 변별력 분석")
        report.append(f"*   **최저-최고 레벨 격차**: {gap}%p (Lv.1 {analysis['min_acc']}% vs Max {analysis['max_acc']}%)")
        if gap < 5:
            report.append("*   **분석 결과**: 레벨 간 지능 차이가 거의 없습니다. 사용자가 난이도를 선택하는 재미가 결여되어 있습니다. 초기 모델의 지능을 강제로 낮춰야 합니다.")
        elif gap < 15:
            report.append("*   **분석 결과**: 어느 정도의 차이는 존재하나, '압도적인 초보'와 '경이로운 전문가' 사이의 체감 난이도 구성에는 미흡합니다.")
        else:
            report.append("*   **분석 결과**: 단계별 지능 차이가 뚜렷하여 게이미피케이션 목적에 부합하는 환경이 조성되었습니다.")

        # 성능 정체 분석
        if analysis['stagnation']:
            report.append(f"\n### ⚠️ 성능 정체 및 역전 현상 발견")
            report.append(f"*   **해당 레벨**: {', '.join(map(str, analysis['stagnation']))}")
            report.append("*   **분석 결과**: 학습 데이터가 증가함에도 성능이 정체되거나 하락하는 구간이 존재합니다. 모델의 수용량(Capacity) 한계이거나 데이터 중복에 의한 과적합 가능성이 큽니다.")
        else:
            report.append("\n### ✅ 안정적인 학습 곡선 유지")
            report.append("*   **분석 결과**: 레벨이 올라감에 따라 성능이 일관되게 향상되고 있습니다. 데이터 증설 전략이 유효하게 작동 중입니다.")

        # 에셋 분석
        report.append("\n### 🔍 에셋별 인지 능력 차이")
        weak_str = ", ".join([f"{v['display_name']}({v['accuracy']}%)" for k, v in analysis['weak_assets']])
        report.append(f"*   **취약 에셋**: {weak_str}")
        report.append("*   **원인 추정**: 복잡한 외곽선이나 미세한 뎁스 차이를 가진 사물에서 AI의 인지 실패가 집중되고 있습니다.")

        report.append("\n---")
        report.append("\n## 3. 종합 진단 및 개선 과제")
        
        diagnosis = "적합" if analysis['is_effective'] else "보완 필요"
        if gap < 5: diagnosis = "부적합"
        
        report.append(f"\n**현재 모델 상태: \"{diagnosis}\"**")
        
        if diagnosis == "부적합":
            report.append("\n현재 훈련된 모델들은 **독립적인 난이도 계층을 형성하는 데 실패**했습니다. 고지능 모델들만 양산되었으며, 이는 사용자가 체감할 수 있는 '성장하는 AI'를 제공하지 못합니다.")
        elif diagnosis == "보완 필요":
            report.append("\n단계별 지능 차이는 존재하나, 실질적인 서비스 경쟁력을 위해 저레벨 모델의 '실수'를 유도하거나 고레벨 모델의 '초정밀함'을 더 강화해야 합니다.")
        else:
            report.append("\n데이터량에 따른 지능 분포가 매우 우수합니다. 현재 가중치를 그대로 서비스에 투입해도 무방합니다.")

        report.append("\n### 🛠 향후 액션 플랜")
        report.append("1. **초기 레벨 지능 강제 하향**: Lv.1~3 모델은 학습 에포크를 1회로 제한하거나 해상도를 64x64로 낮추어 학습시키세요.")
        report.append("2. **데이터 격차 확대**: 레벨 간 학습 데이터 개수를 더 극단적으로 차이나게 배치(예: 10장 vs 1000장) 하세요.")
        report.append("3. **에셋 난이도별 최적화**: 인지하기 어려운 복잡한 에셋을 고레벨 모델에서만 완벽히 학습하도록 조정하세요.")

        report.append("\n\n---\n*본 보고서는 TestReportGenerator에 의해 데이터 분석 기반으로 자동 생성되었습니다.*")
        return "\n".join(report)

    def _generate_llm_report(self, data: Dict[str, Any]) -> str:
        """LLM을 사용하여 더 통찰력 있는 분석 리포트를 생성합니다."""
        analysis = self._analyze_data(data)
        from langchain_core.prompts import ChatPromptTemplate
        
        m_type = data["model_type"].lower()
        timestamp = data.get("timestamp", "unknown")
        img_filename = f"test_accuracy_summary_{m_type}_{timestamp}.png"

        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 AI 모델 성능 분석 전문가이자 게임 기획자입니다. 테스트 데이터를 바탕으로 비판적이고 통찰력 있는 마크다운 보고서를 작성하세요."),
            ("user", """
다음은 'StereoVision Showdown' 프로젝트의 AI 모델 테스트 데이터입니다. 
본 프로젝트의 목표는 '데이터량에 따른 단계별 지능 지수(Lv.1~10) 구현'과 '게이미피케이션'입니다.

[테스트 정보]
- 모델 유형: {model_type}
- 테스트 장치: {device}

[데이터 분석 결과]
- 최저-최고 레벨 격차: {gap}%p
- 성능 정체 구간: {stagnation}
- 취약 에셋: {weak}

[필수 포함 항목]
1. 모델 성능 데이터 요약 (표 형식)
2. 프로젝트 목표 기반 핵심 평가 (Critical Review) - 변별력 상실 문제 등을 신랄하게 지적할 것.
3. 종합 진단 및 향후 액션 플랜 - 게임 서비스로서의 가치를 판단하고 기술적 대안을 제시할 것.

작성 언어: 한국어
            """)
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "model_type": data["model_type"],
            "device": data["device"],
            "gap": analysis["gap"],
            "stagnation": analysis["stagnation"] if analysis["stagnation"] else "없음",
            "weak": ", ".join([f"{v['display_name']}" for k, v in analysis['weak_assets']])
        })
        
        # LLM 응답 본문 상단에 제목과 이미지를 추가하여 리포트 완성
        header = f"# 🧪 StereoVision Showdown: AI 모델 성능 분석 리포트 ({data['model_type'].upper()})\n"
        header += f"**리포트 생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        header += f"![Performance Summary]({img_filename})\n\n"
        
        return header + response.content

    def generate_from_latest_file(self):
        """기존 방식 호환용: 최신 로그 파일을 찾아 리포트 생성"""
        # 최신 파일 찾기 로직 (생략 가능하나 기존 호환 위해 유지)
        pass

if __name__ == "__main__":
    # 독립 실행 시 최신 결과 분석
    generator = TestReportGenerator()
    # 최신 로그 파일 로드 로직 추가 필요
    print("💡 이 유틸리티는 이제 ModelTester에서 직접 호출되거나 최신 로그를 분석합니다.")
