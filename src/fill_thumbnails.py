import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

def extract_thumb(url: str, timeout=10):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        og = soup.find("meta", attrs={"property": "og:image"})
        tw = soup.find("meta", attrs={"name": "twitter:image"})
        if og and og.get("content"):
            return og["content"].strip()
        if tw and tw.get("content"):
            return tw["content"].strip()
        return None
    except Exception:
        return None

def main(run_dir: Path):
    cur_path = run_dir / "curations.json"
    cur = json.loads(cur_path.read_text(encoding="utf-8"))

    updated = 0
    tried = 0
    for theme, items in cur.items():
        for it in items:
            if it.get("thumbnail_url"):
                continue
            url = it.get("url")
            if not url:
                continue
            tried += 1
            thumb = extract_thumb(url)
            if thumb:
                it["thumbnail_url"] = thumb
                updated += 1

    cur_path.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE. tried:", tried, "updated:", updated)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    main(Path(args.run_dir))
