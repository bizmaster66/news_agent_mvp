import json
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup


UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TIMEOUT = 12


def extract_text(html: str, max_chars: int = 8000) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # 너무 길면 앞부분만(요약에 충분)
    return text[:max_chars]


def crawl(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        # html이 아닐 수도 있으니 체크(간단)
        ct = (r.headers.get("content-type") or "").lower()
        if "text/html" not in ct and "application/xhtml" not in ct:
            # 그래도 text로 처리해보고 싶으면 주석 해제 가능
            # pass
            return None
        return extract_text(r.text)
    except Exception:
        return None


def main(run_dir: Path, sleep_sec: float = 0.15):
    cur_path = run_dir / "curations.json"
    cur = json.loads(cur_path.read_text(encoding="utf-8"))

    updated = 0
    tried = 0
    failed = 0

    # Top15만 처리
    for theme, items in cur.items():
        for it in items[:15]:
            # 이미 content_text가 있으면 스킵
            if (it.get("content_text") or "").strip():
                continue

            url = (it.get("url") or "").strip()
            if not url:
                continue

            tried += 1
            text = crawl(url)
            if text:
                it["content_text"] = text
                updated += 1
            else:
                failed += 1

            time.sleep(sleep_sec)

    cur_path.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE. tried:", tried, "updated:", updated, "failed:", failed)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args()

    main(Path(args.run_dir), sleep_sec=args.sleep)
