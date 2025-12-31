import json
import time
from pathlib import Path

from src.gemini_summarizer import make_client, summarize_text, summarize_theme_10_lines

def main(run_dir: Path, sleep_sec: float = 0.2, force: bool = False):
    cur_path = run_dir / "curations.json"
    theme_sum_path = run_dir / "theme_summaries.json"

    cur = json.loads(cur_path.read_text(encoding="utf-8"))
    client = make_client()

    theme_summaries = {}

    for theme, items in cur.items():
        enriched = []

        for it in items[:15]:
            title = it.get("title", "")
            desc = (it.get("description") or "").strip()
            content_text = (it.get("content_text") or "").strip()

            if (not force) and it.get("summary_text"):
                pass
            else:
                out = summarize_text(client, title=title, description=desc, content_text=content_text)
                it["summary_text"] = (out.get("summary_text") or "").strip()
                it["keywords"] = out.get("keywords", [])

            enriched.append({
                "source": it.get("source_name", ""),
                "title": title,
                "summary_text": it.get("summary_text", "")
            })

            time.sleep(sleep_sec)

        # 테마 10줄 요약
        t_out = summarize_theme_10_lines(client, theme=theme, items=enriched)
        theme_summaries[theme] = {"lines": t_out.get("lines", [])}

        # 중간 저장
        cur_path.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        theme_sum_path.write_text(json.dumps(theme_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DONE: summaries generated")
    client.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(Path(args.run_dir), sleep_sec=args.sleep, force=args.force)
