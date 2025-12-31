import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


GROUP_PRIORITY = {"startup": 1, "it": 2, "econ": 3, "daily": 4}


def strip_source_suffix(title: str) -> str:
    t = (title or "").strip()
    for sep in [" - ", " | ", " — ", " – "]:
        if sep in t:
            left, right = t.rsplit(sep, 1)
            if len(right.strip()) <= 25:
                t = left.strip()
    return t


def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("[단독]", "").replace("(단독)", "")
    return s


def build_doc(a: dict) -> str:
    title = clean_text(strip_source_suffix(a.get("title", "")))
    desc = clean_text(a.get("description", ""))
    return f"{title}\n{desc}"


def parse_dt(ts: str):
    if not ts:
        return datetime(1970, 1, 1)
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return datetime(1970, 1, 1)


def pick_representative(items: List[dict]) -> dict:
    def group_rank(a):
        return GROUP_PRIORITY.get(a.get("source_group"), 99)

    def crawl_rank(a):
        return 0 if a.get("crawl_status") == "ok" else 1

    def time_rank(a):
        return parse_dt(a.get("pubdate_ts"))

    # group 우선 → crawl ok → 최신
    return sorted(items, key=lambda x: (group_rank(x), crawl_rank(x), time_rank(x)), reverse=False)[0]


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def dedup_by_similarity(theme_items: List[dict], sim_threshold: float = 0.60, k_neighbors: int = 20):
    """
    테마 후보군 내부에서만 dedup.
    - analyzer='char'로 한국어에서 유사도 신호를 조금 더 강하게 잡음
    - threshold 기본 0.60 (MVP 시작점)
    """
    n = len(theme_items)
    if n <= 1:
        # 각자 클러스터
        return {i: [i] for i in range(n)}

    docs = [build_doc(a) for a in theme_items]

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 6),
        min_df=1,
        max_df=0.98,
    )
    X = vec.fit_transform(docs)

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(k_neighbors, n))
    nn.fit(X)
    dists, inds = nn.kneighbors(X, return_distance=True)

    uf = UnionFind(n)
    for i in range(n):
        for dist, j in zip(dists[i], inds[i]):
            if j == i:
                continue
            sim = 1.0 - float(dist)
            if sim >= sim_threshold:
                uf.union(i, int(j))

    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)
    return clusters


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(run_dir: Path, sim_threshold: float = 0.60, k_neighbors: int = 20, candidate_cap: int = 80):
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    settings = run_json.get("settings_snapshot", {})
    themes = [t for t in settings.get("themes", []) if t.get("enabled", True)]

    articles = load_jsonl(run_dir / "articles.jsonl")

    # 테마별 후보군 뽑기: matched_themes에 테마가 포함된 기사 + rule_scores 기반 정렬
    theme_to_candidates: Dict[str, List[dict]] = {}
    for t in themes:
        tname = t.get("name")
        max_items = int(t.get("max_items", 15))

        cand = []

        must = t.get("must_include", []) or []
        must_any = t.get("must_include_any", []) or []
        for a in articles:
            if must:
                text = ((a.get("title") or "") + " " + (a.get("description") or "")).lower()
                ok_must = all((m or "").lower() in text for m in must)
                if not ok_must:
                    continue
            if must_any:
                text2 = ((a.get("title") or "") + " " + (a.get("description") or "")).lower()
                ok_any = any((m or "").lower() in text2 for m in must_any)
                if not ok_any:
                    continue
            if not any(x.get("theme") == tname for x in (a.get("matched_themes") or [])):
                continue
            score = float((a.get("rule_scores") or {}).get(tname, 0.0))
            # 최신성/크롤 성공 보너스(아주 약하게)
            score += (0.15 if a.get("crawl_status") == "ok" else 0.0)
            score += (0.05 if a.get("pubdate_status") == "ok" else 0.0)
            a2 = dict(a)
            a2["_score"] = score
            cand.append(a2)

        # 후보 정렬: score desc + 최신
        cand.sort(key=lambda x: (x["_score"], parse_dt(x.get("pubdate_ts"))), reverse=True)
        cand = cand[:candidate_cap]  # 테마별 후보군 상한

        theme_to_candidates[tname] = cand

    # dedup + Top15 선정
    curations = {}
    for t in themes:
        tname = t.get("name")
        max_items = int(t.get("max_items", 15))
        cand = theme_to_candidates.get(tname, [])

        if not cand:
            curations[tname] = []
            continue

        clusters = dedup_by_similarity(cand, sim_threshold=sim_threshold, k_neighbors=k_neighbors)

        # 클러스터별 대표기사 선택
        reps = []
        for root, idxs in clusters.items():
            items = [cand[i] for i in idxs]
            rep = pick_representative(items)
            reps.append(rep)

        # 대표기사들을 다시 점수순으로 정렬해서 Top15
        reps.sort(key=lambda x: (x["_score"], parse_dt(x.get("pubdate_ts"))), reverse=True)
        reps = reps[:max_items]

        # 저장 포맷
        out = []
        for rank, r in enumerate(reps, start=1):
            out.append({
                "rank": rank,
                "source_name": r.get("source_name"),
                "source_group": r.get("source_group"),
                "title": r.get("title"),
                "url": r.get("final_url") or r.get("resolved_url") or r.get("google_news_url"),
                "thumbnail_url": r.get("thumbnail_url"),
                "pubdate_ts": r.get("pubdate_ts") or r.get("pubdate_raw"),
                "score": r.get("_score", 0.0),
            })
        curations[tname] = out

    (run_dir / "curations.json").write_text(json.dumps(curations, ensure_ascii=False, indent=2), encoding="utf-8")
    print("CURATION DONE")
    print({k: len(v) for k, v in curations.items()})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--th", type=float, default=0.60)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--cap", type=int, default=80)
    args = ap.parse_args()

    main(Path(args.run_dir), sim_threshold=args.th, k_neighbors=args.k, candidate_cap=args.cap)
