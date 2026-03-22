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
        self.prompt_dir = BASE_DIR / "src" / "prompts"
        
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

    def _load_prompt(self, filename: str) -> Dict[str, str]:
        """외부 마크다운 파일에서 프롬프트 정보를 불러와 파싱합니다."""
        prompt_path = self.prompt_dir / filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # # System Message 및 # User Message 섹션 분리
        system_msg = ""
        user_msg = ""
        
        if "# System Message" in content and "# User Message" in content:
            parts = content.split("# User Message")
            system_msg = parts[0].replace("# System Message", "").strip()
            user_msg = parts[1].strip()
        else:
            user_msg = content.strip()
            
        return {"system": system_msg, "user": user_msg}

    def _get_report_header(self, model_type: str, timestamp: str, img_filename: str) -> str:
        """리포트 공통 헤더를 생성합니다."""
        header = [
            f"# 🧪 StereoVision Showdown: AI 모델 성능 분석 리포트 ({model_type.upper()})",
            f"**리포트 생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**테스트 대상**: AI Player Level 1 ~ 10 ({model_type.upper()} 기반)",
            f"\n![Performance Summary]({img_filename})\n"
        ]
        return "\n".join(header)

    def _generate_rule_based_report(self, data: Dict[str, Any]) -> str:
        """규칙 기반의 비판적 분석 리포트를 생성합니다."""
        analysis = self._analyze_data(data)
        m_type = data["model_type"].lower()
        timestamp = data.get("timestamp", "unknown")
        img_filename = f"test_accuracy_summary_{m_type}_{timestamp}.png"
        
        # 1. 요약 테이블 생성
        summary_rows = []
        for r in data["results"]:
            lv = r['level']
            target = "Beginner" if lv <= 3 else "Intermediate" if lv <= 7 else "Expert"
            status = "🟢 적합" if r['accuracy'] > (lv * 5 + 50) else "🟡 미흡"
            summary_rows.append(f"| **Lv.{lv}** | {r['accuracy']}% | {r['total_images']} | {target} 모델 | {status} |")
        summary_table = "\n".join(summary_rows)

        # 2. 변별력 분석 데이터 준비
        gap = analysis['gap']
        discrimination_icon = '✅' if gap > 15 else '⚠️ [주의]' if gap > 5 else '❌ [심각]'
        
        if gap < 5:
            discrimination_result = "레벨 간 지능 차이가 거의 없습니다. 사용자가 난이도를 선택하는 재미가 결여되어 있습니다. 초기 모델의 지능을 강제로 낮춰야 합니다."
        elif gap < 15:
            discrimination_result = "어느 정도의 차이는 존재하나, '압도적인 초보'와 '경이로운 전문가' 사이의 체감 난이도 구성에는 미흡합니다."
        else:
            discrimination_result = "단계별 지능 차이가 뚜렷하여 게이미피케이션 목적에 부합하는 환경이 조성되었습니다."

        # 3. 성능 정체 분석 데이터 준비
        if analysis['stagnation']:
            stagnation_section = "### ⚠️ 성능 정체 및 역전 현상 발견\n"
            stagnation_section += f"*   **해당 레벨**: {', '.join(map(str, analysis['stagnation']))}\n"
            stagnation_section += "*   **분석 결과**: 학습 데이터가 증가함에도 성능이 정체되거나 하락하는 구간이 존재합니다. 모델의 수용량(Capacity) 한계이거나 데이터 중복에 의한 과적합 가능성이 큽니다."
        else:
            stagnation_section = "### ✅ 안정적인 학습 곡선 유지\n"
            stagnation_section += "*   **분석 결과**: 레벨이 올라감에 따라 성능이 일관되게 향상되고 있습니다. 데이터 증설 전략이 유효하게 작동 중입니다."

        # 4. 취약 에셋 문자열
        weak_assets_str = ", ".join([f"{v['display_name']}({v['accuracy']}%)" for k, v in analysis['weak_assets']])

        # 5. 종합 진단 데이터 준비
        diagnosis = "적합" if analysis['is_effective'] else "보완 필요"
        if gap < 5: diagnosis = "부적합"
        
        if diagnosis == "부적합":
            diagnosis_detail = "현재 훈련된 모델들은 **독립적인 난이도 계층을 형성하는 데 실패**했습니다. 고지능 모델들만 양산되었으며, 이는 사용자가 체감할 수 있는 '성장하는 AI'를 제공하지 못합니다."
        elif diagnosis == "보완 필요":
            diagnosis_detail = "단계별 지능 차이는 존재하나, 실질적인 서비스 경쟁력을 위해 저레벨 모델의 '실수'를 유도하거나 고레벨 모델의 '초정밀함'을 더 강화해야 합니다."
        else:
            diagnosis_detail = "데이터량에 따른 지능 분포가 매우 우수합니다. 현재 가중치를 그대로 서비스에 투입해도 무방합니다."

        # 템플릿 로드 및 데이터 채우기
        template_data = self._load_prompt("test_report_rule_template.md")
        template = template_data["user"] # 규칙 기반은 user 섹션 전체를 템플릿으로 사용
        
        report_content = template.format(
            model_type_upper=m_type.upper(),
            report_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            device=data.get('device', 'unknown'),
            img_filename=img_filename,
            summary_table=summary_table,
            discrimination_icon=discrimination_icon,
            gap=gap,
            min_acc=analysis['min_acc'],
            max_acc=analysis['max_acc'],
            discrimination_result=discrimination_result,
            stagnation_section=stagnation_section,
            weak_assets_str=weak_assets_str,
            diagnosis=diagnosis,
            diagnosis_detail=diagnosis_detail
        )
        
        return report_content

    def _generate_llm_report(self, data: Dict[str, Any]) -> str:
        """LLM을 사용하여 더 통찰력 있는 분석 리포트를 생성합니다."""
        analysis = self._analyze_data(data)
        from langchain_core.prompts import ChatPromptTemplate
        
        m_type = data["model_type"].lower()
        timestamp = data.get("timestamp", "unknown")
        img_filename = f"test_accuracy_summary_{m_type}_{timestamp}.png"

        prompt_data = self._load_prompt("test_report_llm_prompt.md")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_data["system"]),
            ("user", prompt_data["user"])
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "model_type": data["model_type"],
            "device": data["device"],
            "gap": analysis["gap"],
            "stagnation": analysis["stagnation"] if analysis["stagnation"] else "없음",
            "weak": ", ".join([f"{v['display_name']}" for k, v in analysis['weak_assets']])
        })
        
        header = self._get_report_header(data['model_type'], timestamp, img_filename)
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
