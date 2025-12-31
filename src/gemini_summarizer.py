import json
import re
from typing import Any, Dict

from google import genai
from google.genai import types

MODEL_ID = "gemini-2.5-flash"

def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty response text")
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON object found in: {text[:200]}")
    return json.loads(m.group(0))

def make_client() -> genai.Client:
    return genai.Client()

def summarize_text(client: genai.Client, title: str, description: str, content_text: str) -> Dict[str, Any]:
    """
    기사 1개 -> 줄글 요약(JSON)
    - description + (가능하면) content_text를 근거로 더 구체화
    """
    content_text = (content_text or "").strip()
    # 너무 길면 요약 입력 상한(비용/속도)
    if len(content_text) > 5000:
        content_text = content_text[:5000]

    prompt = f"""
너는 뉴스 요약 에이전트다. 아래 기사 정보를 근거로 "구체적인 사실" 중심으로 요약하라.
반드시 JSON만 출력하라.

[기사]
제목: {title}
요약(description): {description}

[원문 발췌(content_text, 있을 경우)]
{content_text if content_text else "(원문 발췌 없음)"}

[출력 JSON 스키마]
{{
  "summary_text": "2~3문장 줄글 요약",
  "keywords": ["키워드1", "키워드2", "키워드3"]
}}

규칙(중요):
- summary_text는 2~3문장, 한국어, 줄글 형태.
- 숫자/규모/대상/주체/행동(지원·출시·투자 등)이 있으면 반드시 포함.
- 기사에 없는 내용은 추측 금지.
- '정보 부족', '파악하기 어렵다' 같은 메타 코멘트는 금지. (description이 비어있고 content_text도 없을 때만 예외)
"""
    resp = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.9,
        ),
    )
    return _extract_json(resp.text)

def summarize_theme_10_lines(client: genai.Client, theme: str, items: list) -> Dict[str, Any]:
    packed = []
    for it in items:
        packed.append(f"- {it.get('source','')} | {it.get('title','')}\n  요약: {it.get('summary_text','')}")
    prompt = f"""
너는 뉴스 리포트 작성자다. 아래는 테마 '{theme}'의 상위 기사 요약들이다.
중복 이슈는 묶어서 10줄 인사이트로 정리하라. 반드시 JSON만 출력하라.

[기사 요약들]
{chr(10).join(packed)}

[출력 JSON 스키마]
{{
  "lines": ["1...", "2...", "3...", "4...", "5...", "6...", "7...", "8...", "9...", "10..."]
}}

규칙:
- lines는 반드시 10개.
- 과장 금지, 사실 중심.
"""
    resp = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            top_p=0.9,
        ),
    )
    return _extract_json(resp.text)
