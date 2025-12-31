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


def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    # 흔한 꼬리/표기 제거
    s = s.replace("[단독]", "").replace("(단독)", "")
    return s


def build_doc(a: dict) -> str:
    """
    본문이 있으면 본문 일부를 우선 사용하고,
    없으면 title + description 기반으로 유사도 계산.
    """
    title = clean_text(a.get("title", ""))
    desc = clean_text(a.get("description", ""))
    body = a.get("content_text") or ""

    # 본문이 있으면 앞부분만 섞어주면 유사도 안정성↑
    if body:
        body = re.sub(r"\s+", " ", body).strip()
        body = body[:2500]
        return f"{title}\n{desc}\n{body}"
    return f"{title}\n{desc}"


def pick_representative(articles: List[dict]) -> dict:
    """
    대표기사 선택 규칙:
    1) source_group 우선순위 (startup > it > econ > daily)
    2) crawl_status ok 우선
    3) pubdate_ts 최신 우선
    """
    def group_rank(a):
        return GROUP_PRIORITY.get(a.get("source_group"), 99)

    def crawl_rank(a):
        return 0 if a.get("crawl_status") == "ok" else 1

    def time_rank(a):
        ts = a.get("pubdate_ts")
        if not ts:
            return datetime(1970, 1, 1)
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return datetime(1970, 1, 1)

    return sorted(articles, key=lambda x: (group_rank(x), crawl_rank(x), time_rank(x)))[0]


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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


def run_dedup(run_dir: Path, sim_threshold: float = 0.83) -> Tuple[int, int, dict]:
    """
    sim_threshold: 높을수록(0.9) 더 보수적으로 묶음, 낮을수록(0.75) 더 많이 묶음
    추천 시작값: 0.83
    """
    articles_path = run_dir / "articles.jsonl"
    if not articles_path.exists():
        raise FileNotFoundError(f"articles.jsonl not found: {articles_path}")

    rows = load_jsonl(articles_path)
    n = len(rows)
    before = n

    docs = [build_doc(a) for a in rows]

    # ✅ 한국어/제목 변형 대응: 문자 n-gram TF-IDF (형태소 없이도 강함)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_df=0.98,
    )
    X = vectorizer.fit_transform(docs)

    # 근접 이웃 기반으로 유사 문서만 탐색 (O(n^2) 피함)
    # cosine distance = 1 - cosine similarity
    nbrs = NearestNeighbors(metric="cosine", algorithm="brute")
    nbrs.fit(X)

    # radius = 1 - threshold
    radius = 1.0 - sim_threshold
    neighborhoods = nbrs.radius_neighbors(X, radius=radius, return_distance=True)

    uf = UnionFind(n)

    # 유사한 것끼리 union
    distances_list, indices_list = neighborhoods
    links = 0
    for i in range(n):
        inds = indices_list[i]
        dists = distances_list[i]
        for j, dist in zip(inds, dists):
            if j == i:
                continue
            sim = 1.0 - float(dist)
            if sim >= sim_threshold:
                uf.union(i, int(j))
                links += 1

    # cluster 구성
    cluster_map: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        cluster_map[uf.find(i)].append(i)

    # 대표기사 선택 및 마킹
    reps = 0
    cluster_items = list(cluster_map.items())

    for ci, (_, idxs) in enumerate(cluster_items, start=1):
        cluster_id = f"c-{ci:05d}"
        group = [rows[i] for i in idxs]
        rep = pick_representative(group)

        # reset flags
        for i in idxs:
            rows[i]["dedup_cluster_id"] = cluster_id
            rows[i]["is_representative"] = False
            # 본문이 없는 비중이 높으므로 weak 플래그는 “본문 없는 상태에서 묶였는지”로 표시
            rows[i]["dedup_weak"] = (rows[i].get("content_text") is None)

        # representative marking (index match)
        rep_idx = None
        for i in idxs:
            if rows[i] is rep:
                rep_idx = i
                break
        if rep_idx is None:
            # 안전하게 title+url로 찾아보기
            rep_title = rep.get("title", "")
            rep_url = rep.get("final_url") or rep.get("resolved_url") or rep.get("google_news_url") or ""
            for i in idxs:
                url = rows[i].get("final_url") or rows[i].get("resolved_url") or rows[i].get("google_news_url") or ""
                if rows[i].get("title", "") == rep_title and url == rep_url:
                    rep_idx = i
                    break

        if rep_idx is None:
            rep_idx = idxs[0]

        rows[rep_idx]["is_representative"] = True
        reps += 1

    write_jsonl(articles_path, rows)

    after = reps
    stats = {
        "before": before,
        "after": after,
        "clusters": len(cluster_items),
        "representatives": reps,
        "similarity_threshold": sim_threshold,
        "radius_neighbors_links": links,
    }
    return before, after, stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="data/runs/<run_folder> 경로")
    ap.add_argument("--th", type=float, default=0.83, help="similarity threshold (0~1)")
    args = ap.parse_args()

    before, after, stats = run_dedup(Path(args.run_dir), sim_threshold=args.th)
    print("DEDUP DONE")
    print(stats)
