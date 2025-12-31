import re
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from dateutil import tz

KST = tz.gettz("Asia/Seoul")

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

DEFAULT_TIMEOUT = 10


@dataclass
class RawArticle:
    title: str
    google_news_url: str
    description: str
    pubdate_raw: str
    pubdate_ts: Optional[datetime]
    pubdate_status: str  # ok/unknown
    source_domain: str
    source_name: str
    source_group: str

    resolved_url: Optional[str] = None
    final_url: Optional[str] = None
    canonical_url: Optional[str] = None
    crawl_status: str = "pending"  # ok/failed/skipped/pending
    thumbnail_url: Optional[str] = None
    content_text: Optional[str] = None

    matched_themes: Optional[List[Dict]] = None
    rule_scores: Optional[Dict[str, float]] = None


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s


def _normalize_text(s: str) -> str:
    s = (s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def _expand_kw(kw: str) -> List[str]:
    """
    'VC/PE/AC/신기사' 같은 표현은 / 기준으로 분해해 OR 키워드로 사용
    """
    kw = (kw or "").strip()
    if not kw:
        return []
    if "/" in kw:
        parts = [p.strip() for p in kw.split("/") if p.strip()]
        return parts if parts else [kw]
    return [kw]


def build_union_keywords(settings: dict) -> List[str]:
    kws = set()
    for t in settings.get("themes", []):
        if not t.get("enabled", True):
            continue
        for grp in t.get("include_groups", []) or []:
            for kw in grp or []:
                for sub in _expand_kw(kw):
                    sub = (sub or "").strip()
                    if sub:
                        kws.add(sub)
    # 너무 짧은 단어는 잡음이 많아 제외
    kws = {k for k in kws if len(k) >= 2}
    return sorted(kws)


def build_google_news_rss_url(domain: str, union_keywords: List[str]) -> str:
    if not union_keywords:
        q = f"site:{domain}"
    else:
        MAX_K = 25
        ks = union_keywords[:MAX_K]
        or_part = " OR ".join([f'"{k}"' if " " in k else k for k in ks])
        q = f"site:{domain} ({or_part})"

    params = {"q": q, "hl": "ko", "gl": "KR", "ceid": "KR:ko"}
    return "https://news.google.com/rss/search?" + urllib.parse.urlencode(params)


def parse_pubdate(entry) -> Tuple[str, Optional[datetime], str]:
    raw = ""
    ts = None
    status = "unknown"
    try:
        raw = entry.get("published", "") or entry.get("pubDate", "") or ""
        if raw:
            dt = dtparser.parse(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=KST)
            ts = dt.astimezone(KST)
            status = "ok"
    except Exception:
        status = "unknown"
    return raw, ts, status


def fetch_rss_for_media(session: requests.Session, media: dict, settings: dict, sleep_sec: float = 0.15) -> List[RawArticle]:
    domain = media["domain"]
    url = build_google_news_rss_url(domain, build_union_keywords(settings))

    # ✅ 핵심: feedparser가 직접 URL을 열지 말고, requests로 UA 붙여서 가져온 뒤 parse
    try:
        r = session.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        d = feedparser.parse(r.content)
    except Exception:
        d = feedparser.FeedParserDict()
        d.entries = []

    items: List[RawArticle] = []
    for e in getattr(d, "entries", []) or []:
        title = (e.get("title") or "").strip()
        g_url = (e.get("link") or "").strip()
        desc = (e.get("summary") or e.get("description") or "").strip()

        pub_raw, pub_ts, pub_status = parse_pubdate(e)

        items.append(
            RawArticle(
                title=title,
                google_news_url=g_url,
                description=desc,
                pubdate_raw=pub_raw,
                pubdate_ts=pub_ts,
                pubdate_status=pub_status,
                source_domain=domain,
                source_name=media.get("name", domain),
                source_group=media.get("group", "econ"),
            )
        )

    time.sleep(sleep_sec)
    return items


def filter_by_time_window(items: List[RawArticle], start_ts: datetime, end_ts: datetime) -> Tuple[List[RawArticle], int]:
    kept: List[RawArticle] = []
    unknown = 0
    for it in items:
        if it.pubdate_status != "ok" or it.pubdate_ts is None:
            kept.append(it)
            unknown += 1
            continue
        if start_ts <= it.pubdate_ts <= end_ts:
            kept.append(it)
    return kept, unknown


def _match_group(text: str, group: List[str]) -> Tuple[bool, List[str]]:
    matched = []
    for kw in group:
        for sub in _expand_kw(kw):
            sub_norm = _normalize_text(sub)
            if sub_norm and sub_norm in text:
                matched.append(sub)
    return (len(matched) > 0), sorted(set(matched))


def rule_match_theme(article_text: str, theme: dict) -> Tuple[bool, List[str], float]:
    text = _normalize_text(article_text)

    for ex in theme.get("exclude_keywords", []) or []:
        for sub in _expand_kw(ex):
            sub_norm = _normalize_text(sub)
            if sub_norm and sub_norm in text:
                return False, [], 0.0

    # must_include: 하나라도 없으면 탈락
    must = theme.get("must_include", []) or []
    for mk in must:
        mk_norm = _normalize_text(mk)
        if mk_norm and mk_norm not in text:
            return False, [], 0.0

    include_groups = theme.get("include_groups", []) or []
    if not include_groups or include_groups == [[]]:
        return False, [], 0.0

    all_matched_kws: List[str] = []
    for grp in include_groups:
        ok, matched = _match_group(text, grp or [])
        if not ok:
            return False, [], 0.0
        all_matched_kws.extend(matched)

    uniq = sorted(set(all_matched_kws))
    score = float(len(uniq))
    return True, uniq, score


def pick_crawl_candidates(items: List[RawArticle], settings: dict) -> Tuple[List[int], Dict[str, List[int]]]:
    themes = [t for t in settings.get("themes", []) if t.get("enabled", True)]
    if not themes:
        return [], {}

    EXTRA = 30
    PER_THEME_CAP = 60
    GLOBAL_CAP = 160

    texts = [f"{it.title}\n{it.description}" for it in items]

    theme_candidates: Dict[str, List[Tuple[int, float, datetime]]] = {}
    theme_to_idxs: Dict[str, List[int]] = {}

    for theme in themes:
        tname = theme.get("name")
        theme_candidates[tname] = []
        for idx, it in enumerate(items):
            ok, matched_kws, base_score = rule_match_theme(texts[idx], theme)
            if not ok:
                continue

            title_text = _normalize_text(it.title)
            title_bonus = 0.0
            for mk in matched_kws:
                if _normalize_text(mk) in title_text:
                    title_bonus += 1.0
            score = base_score + title_bonus * 0.8
            dt = it.pubdate_ts if it.pubdate_ts else datetime(1970, 1, 1, tzinfo=KST)
            theme_candidates[tname].append((idx, score, dt))

    selected_all: List[int] = []
    for tname, cand in theme_candidates.items():
        if not cand:
            theme_to_idxs[tname] = []
            continue
        cand.sort(key=lambda x: (x[1], x[2]), reverse=True)
        theme_obj = next((t for t in themes if t.get("name") == tname), None)
        max_items = int((theme_obj or {}).get("max_items", 15))
        top_n = min(max_items + EXTRA, PER_THEME_CAP)
        idxs = [x[0] for x in cand[:top_n]]
        theme_to_idxs[tname] = idxs
        selected_all.extend(idxs)

    uniq = sorted(set(selected_all))

    if len(uniq) > GLOBAL_CAP:
        scored = []
        for idx in uniq:
            appear = sum(1 for _, idxs in theme_to_idxs.items() if idx in idxs)
            dt = items[idx].pubdate_ts if items[idx].pubdate_ts else datetime(1970, 1, 1, tzinfo=KST)
            scored.append((idx, appear, dt))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        uniq = [x[0] for x in scored[:GLOBAL_CAP]]

    return uniq, theme_to_idxs


def resolve_google_news_url(session: requests.Session, google_url: str) -> Tuple[Optional[str], Optional[str]]:
    if not google_url:
        return None, None
    try:
        r = session.get(google_url, allow_redirects=True, timeout=DEFAULT_TIMEOUT)
        return r.url, r.url
    except Exception:
        return None, None


def extract_og_image_and_canonical(html: str) -> Tuple[Optional[str], Optional[str]]:
    soup = BeautifulSoup(html, "lxml")
    og = soup.find("meta", attrs={"property": "og:image"})
    tw = soup.find("meta", attrs={"name": "twitter:image"})
    canon = soup.find("link", attrs={"rel": "canonical"})

    thumb = None
    if og and og.get("content"):
        thumb = og.get("content").strip()
    elif tw and tw.get("content"):
        thumb = tw.get("content").strip()

    canonical = canon.get("href").strip() if canon and canon.get("href") else None
    return thumb, canonical


def extract_text_simple(html: str, max_chars: int = 9000) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]


def crawl_article(session: requests.Session, url: str) -> Tuple[str, Optional[str], Optional[str]]:
    r = session.get(url, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    html = r.text
    thumb, canonical = extract_og_image_and_canonical(html)
    text = extract_text_simple(html)
    return text, thumb, canonical


def fetch_articles(settings: dict, start_ts: datetime, end_ts: datetime) -> Dict:
    enabled_media = [m for m in settings.get("media_sources", []) if m.get("enabled", True)]

    session = _make_session()

    all_items: List[RawArticle] = []
    for m in enabled_media:
        all_items.extend(fetch_rss_for_media(session, m, settings))

    filtered, pub_unknown = filter_by_time_window(all_items, start_ts, end_ts)

    crawl_idxs, theme_to_idxs = pick_crawl_candidates(filtered, settings)
    crawl_idx_set = set(crawl_idxs)

    stats = {
        "rss_items": len(all_items),
        "after_time_filter": len(filtered),
        "pubdate_unknown": pub_unknown,
        "crawl_candidates": len(crawl_idxs),
        "crawl_ok": 0,
        "crawl_failed": 0,
        "crawl_skipped": 0,
    }

    for idx, it in enumerate(filtered):
        article_text = f"{it.title}\n{it.description}"
        matched = []
        scores = {}
        for th in settings.get("themes", []):
            if not th.get("enabled", True):
                continue
            ok, matched_kws, score = rule_match_theme(article_text, th)
            if ok:
                matched.append({
                    "theme": th.get("name"),
                    "label_source": "rule",
                    "confidence": 0.65,
                    "matched_keywords": matched_kws
                })
                scores[th.get("name")] = score
        it.matched_themes = matched
        it.rule_scores = scores

        if idx not in crawl_idx_set:
            it.crawl_status = "skipped"
            stats["crawl_skipped"] += 1
            continue

        resolved, final_url = resolve_google_news_url(session, it.google_news_url)
        it.resolved_url = resolved
        it.final_url = final_url

        if final_url and "news.google.com" not in final_url:
            try:
                text, thumb, canonical = crawl_article(session, final_url)
                it.content_text = text
                it.thumbnail_url = thumb
                it.canonical_url = canonical
                it.crawl_status = "ok"
                stats["crawl_ok"] += 1
            except Exception:
                it.crawl_status = "failed"
                stats["crawl_failed"] += 1
        else:
            it.crawl_status = "skipped"
            stats["crawl_skipped"] += 1

        time.sleep(0.06)

    articles_json = []
    for it in filtered:
        articles_json.append(
            {
                "title": it.title,
                "google_news_url": it.google_news_url,
                "resolved_url": it.resolved_url,
                "final_url": it.final_url,
                "canonical_url": it.canonical_url,
                "description": it.description,
                "pubdate_raw": it.pubdate_raw,
                "pubdate_ts": it.pubdate_ts.isoformat() if it.pubdate_ts else None,
                "pubdate_status": it.pubdate_status,
                "source_domain": it.source_domain,
                "source_name": it.source_name,
                "source_group": it.source_group,
                "crawl_status": it.crawl_status,
                "thumbnail_url": it.thumbnail_url,
                "content_text": it.content_text,
                "matched_themes": it.matched_themes or [],
                "rule_scores": it.rule_scores or {},
            }
        )

    return {"stats": stats, "articles": articles_json, "theme_to_candidate_indexes": theme_to_idxs}
