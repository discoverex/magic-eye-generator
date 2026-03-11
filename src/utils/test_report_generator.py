import json
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config.settings import BASE_DIR, OPENAI_API_KEY


class TestReportGenerator:
    """
    테스트 결과 JSON을 분석하여 LLM 기반의 상세 마크다운 리포트를 생성하는 유틸리티
    """
    def __init__(self):
        self.result_dir = BASE_DIR / "test_results"
        self.report_path = self.result_dir / "test_report.md"
        
        # LLM 설정 (OpenAI)
        if not OPENAI_API_KEY:
            self.llm = None
        else:
            self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.7)

    def _load_latest_logs(self):
        """가장 최근에 생성된 pth 또는 onnx 로그 파일을 로드합니다."""
        pth_log = self.result_dir / "test_logs_pth.json"
        onnx_log = self.result_dir / "test_logs_onnx.json"
        
        # 파일 수정 시간 기준으로 최신 파일 선택
        logs = []
        if pth_log.exists(): logs.append(pth_log)
        if onnx_log.exists(): logs.append(onnx_log)
        
        if not logs:
            return None
            
        latest_log = max(logs, key=lambda p: os.path.getmtime(p))
        with open(latest_log, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate(self):
        """데이터 분석 및 리포트 생성 메인 로직"""
        print("\n📊 테스트 결과 분석 및 리포트 생성 시작...")
        
        data = self._load_latest_logs()
        if not data:
            print("❌ 분석할 테스트 로그 파일이 없습니다. 먼저 모델 테스트를 실행해 주세요.")
            return

        if not self.llm:
            print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다. 기본 서식으로 리포트를 생성합니다.")
            self._generate_basic_report(data)
            return

        # LLM용 요약 데이터 구성
        summary_stats = []
        for res in data["results"]:
            summary_stats.append({
                "level": res["level"],
                "accuracy": res["accuracy"],
                "total": res["total_images"]
            })

        # 가장 성능이 낮은 에셋과 높은 에셋 파악 (마지막 레벨 기준)
        last_result = data["results"][-1]
        asset_metrics = last_result["per_asset_metrics"]
        sorted_assets = sorted(asset_metrics.items(), key=lambda x: x[1]["accuracy"])
        
        weak_assets = [f"{v['display_name']}({v['accuracy']}%)" for k, v in sorted_assets[:3]]
        strong_assets = [f"{v['display_name']}({v['accuracy']}%)" for k, v in sorted_assets[-3:]]

        # 프롬프트 설계
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 AI 모델 성능 분석 전문가입니다. 주어진 테스트 데이터를 바탕으로 전문적이고 통찰력 있는 마크다운 형식의 리포트를 작성하세요. 한국어로 작성해야 합니다."),
            ("user", """
다음은 'StereoVision Showdown' 프로젝트의 AI 모델 테스트 결과 데이터입니다.

[테스트 정보]
- 모델 유형: {model_type}
- 테스트 시간: {timestamp}
- 테스트 장치: {device}

[레벨별 정확도 현황]
{stats}

[최종 레벨 상세 분석]
- 강점 에셋: {strong}
- 약점 에셋: {weak}

[작성 가이드라인]
1. 테스트 개요와 목적을 서술하세요.
2. 레벨별 성능 추이에 대한 분석을 포함하세요 (성장 곡선 등).
3. 특정 에셋(사물)에 대해 왜 성능 차이가 발생하는지 컴퓨터 비전 관점에서 추측해 보세요.
4. 향후 모델 개선을 위한 제언(데이터 증강, 하이퍼파라미터 조정 등)을 포함하세요.
5. 마크다운 형식을 지키며, 시각화 이미지는 `![Performance](test_accuracy_summary_{model_type}.png)` 형식으로 포함될 것임을 고려하여 작성하세요.
            """)
        ])

        chain = prompt | self.llm
        
        try:
            response = chain.invoke({
                "model_type": data["model_type"],
                "timestamp": data["test_timestamp"],
                "device": data["device"],
                "stats": json.dumps(summary_stats, indent=2),
                "strong": ", ".join(strong_assets),
                "weak": ", ".join(weak_assets)
            })
            
            self._save_report(response.content)
            print(f"✅ AI 분석 리포트 생성 완료: {self.report_path}")
        except Exception as e:
            print(f"❌ LLM 리포트 생성 중 에러 발생: {e}")
            self._generate_basic_report(data)

    def _generate_basic_report(self, data):
        """LLM을 사용할 수 없을 때 생성하는 기본 마크다운 리포트"""
        m_type = data["model_type"]
        content = f"""# 📊 AI 모델 성능 테스트 리포트 ({m_type.upper()})

## 1. 테스트 개요
- **일시**: {data['test_timestamp']}
- **대상 모델**: {m_type.upper()} 기반 AI Player (1~10단계)
- **테스트 환경**: {data['device']}

## 2. 성능 시각화
![Performance Summary](test_accuracy_summary_{m_type}.png)

## 3. 레벨별 정확도 요약
| AI 레벨 | 테스트 이미지 수 | 정답 수 | 정확도 |
|:---:|:---:|:---:|:---:|
"""
        for res in data["results"]:
            content += f"| Level {res['level']} | {res['total_images']} | {res['correct_images']} | {res['accuracy']}% |\n"
            
        content += "\n\n> 💡 상세 분석을 보려면 OPENAI_API_KEY를 설정하고 리포트를 다시 생성하세요."
        
        self._save_report(content)
        print(f"✅ 기본 리포트 생성 완료: {self.report_path}")

    def _save_report(self, content):
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(content)

if __name__ == "__main__":
    generator = TestReportGenerator()
    generator.generate()
