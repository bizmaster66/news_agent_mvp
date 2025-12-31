import json
import shutil
import sys
import uuid
from datetime import datetime, timedelta, time
from pathlib import Path

import streamlit as st
from dateutil import tz

ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

from src.news_fetcher import fetch_articles  # noqa: E402

DATA_DIR = ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
SETTINGS_PATH = DATA_DIR / "settings_current.json"

KST = tz.gettz("Asia/Seoul")


def now_kst() -> datetime:
    return datetime.now(tz=KST)


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


def save_settings(settings: dict):
    settings["updated_at"] = now_kst().isoformat()
    SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def get_default_window(settings: dict):
    win = settings.get("default_run_window", {})
    start_s = win.get("start_time", "08:00:00")
    end_s = win.get("end_time", "07:59:59")

    def parse_hms(hms: str) -> time:
        hh, mm, ss = [int(x) for x in hms.split(":")]
        return time(hh, mm, ss)

    today = now_kst().date()
    start_dt = datetime.combine(today - timedelta(days=1), parse_hms(start_s), tzinfo=KST)
    end_dt = datetime.combine(today, parse_hms(end_s), tzinfo=KST)
    return start_dt, end_dt


def safe_slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


def make_run_folder(start_dt: datetime, end_dt: datetime) -> Path:
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    folder_name = f"{start_dt.strftime('%Y-%m-%d_%H%M%S')}__{end_dt.strftime('%Y-%m-%d_%H%M%S')}__{run_id}"
    path = RUNS_DIR / safe_slug(folder_name)
    path.mkdir(parents=True, exist_ok=False)
    return path


def list_runs():
    if not RUNS_DIR.exists():
        return []
    runs = [p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "run.json").exists()]
    runs.sort(key=lambda x: x.name, reverse=True)
    return runs


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_run_folder(run_path: Path):
    shutil.rmtree(run_path)


st.set_page_config(page_title="News Agent MVP", layout="wide")
ensure_dirs()

st.sidebar.title("News Agent MVP")
page = st.sidebar.radio("메뉴", ["Run", "Result", "History", "Settings"])

settings = load_settings()


# -----------------------------
# Page: Run
# -----------------------------
if page == "Run":
    st.header("Run: 기간을 선택하고 실행")

    if not settings:
        st.error("data/settings_current.json을 찾지 못했습니다.")
        st.stop()

    default_start, default_end = get_default_window(settings)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("시작 날짜 (KST)", value=default_start.date())
        start_time = st.time_input("시작 시간 (KST)", value=default_start.timetz().replace(tzinfo=None))
    with col2:
        end_date = st.date_input("종료 날짜 (KST)", value=default_end.date())
        end_time = st.time_input("종료 시간 (KST)", value=default_end.timetz().replace(tzinfo=None))

    start_dt = datetime.combine(start_date, start_time, tzinfo=KST)
    end_dt = datetime.combine(end_date, end_time, tzinfo=KST)

    if end_dt <= start_dt:
        st.warning("종료 시간이 시작 시간보다 이후여야 합니다.")
        st.stop()

    st.caption("MVP: RSS 수집 → 기간 필터 → (후보만) 원문 크롤링 → articles.jsonl 저장")

    if st.button("실행 (RSS 수집 & 저장)"):
        run_path = make_run_folder(start_dt, end_dt)

        with st.spinner("RSS 수집 및 처리 중..."):
            result = fetch_articles(settings, start_dt, end_dt)

        stats = dict(result.get("stats", {}))
        stats.setdefault("dedup_before", stats.get("after_time_filter", 0))
        stats.setdefault("dedup_after", stats.get("after_time_filter", 0))

        run_meta = {
            "run_id": run_path.name.split("__")[-1],
            "start_ts": start_dt.isoformat(),
            "end_ts": end_dt.isoformat(),
            "created_at": now_kst().isoformat(),
            "status": "succeeded",
            "settings_snapshot": settings,
            "stats": stats,
        }
        write_json(run_path / "run.json", run_meta)

        # articles.jsonl 저장
        with open(run_path / "articles.jsonl", "w", encoding="utf-8") as f:
            for a in result.get("articles", []):
                f.write(json.dumps(a, ensure_ascii=False) + "\n")

        # 결과 파일 기본 생성(없으면)
        if not (run_path / "curations.json").exists():
            write_json(run_path / "curations.json", {})
        if not (run_path / "theme_summaries.json").exists():
            write_json(run_path / "theme_summaries.json", {})

        st.success(f"수집 완료: {len(result.get('articles', []))}개 기사 저장")
        st.info(f"Run: {run_path.name}")
        st.info("Result 탭에서 확인하세요.")


# -----------------------------
# Page: Result
# -----------------------------
elif page == "Result":
    st.header("Result: 실행 결과 보기")

    runs = list_runs()
    if not runs:
        st.info("아직 생성된 Run이 없습니다. Run 탭에서 실행을 먼저 해주세요.")
        st.stop()

    run_names = [p.name for p in runs]
    selected = st.selectbox("Run 선택", run_names, index=0)
    run_path = RUNS_DIR / selected

    run_meta = read_json(run_path / "run.json")
    st.subheader("Run 정보")
    st.json(
        {
            "run_id": run_meta.get("run_id"),
            "period": [run_meta.get("start_ts"), run_meta.get("end_ts")],
            "created_at": run_meta.get("created_at"),
            "status": run_meta.get("status"),
            "stats": run_meta.get("stats", {}),
        }
    )

    st.divider()

    # ---- Actions: curate / summarize (cloud-safe direct call) ----
    st.subheader("Actions")

    import io, contextlib, traceback
    from pathlib import Path
    from src import curate_rule, summarize_top15

    def run_and_show(label, fn):
        st.toast(label)
        buf = io.StringIO()
        with st.spinner(label):
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    fn()
            except Exception:
                traceback.print_exc(file=buf)
        out = buf.getvalue().strip()
        if out:
            st.code(out)
        else:
            st.info("완료(출력 없음)")
        st.rerun()

    c1, c2, c3 = st.columns([2,2,6])
    with c1:
        if st.button("큐레이션 생성 (curate)", key=f"curate_{run_path.name}"):
            run_and_show(
                "큐레이션 생성 중...",
                lambda: curate_rule.main(Path(run_path), sim_threshold=0.60, k_neighbors=20, candidate_cap=80)
            )
    with c2:
        if st.button("요약 생성 (force)", key=f"sum_{run_path.name}"):
            run_and_show(
                "요약 생성 중...",
                lambda: summarize_top15.main(Path(run_path), sleep_sec=0.2, force=True)
            )
    with c3:
        st.caption("※ Settings 변경 후, 같은 Run에 반영하려면 curate → summarize를 다시 실행하세요.")


    # ---- Actions: curate / summarize (no terminal) ----
    

st.subheader("수집 기사 (일부 30개 미리보기)")
    articles_file = run_path / "articles.jsonl"
    if articles_file.exists():
        rows = []
        with open(articles_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 30:
                    break
                if line.strip():
                    rows.append(json.loads(line))

        if not rows:
            st.info("articles.jsonl이 비어 있습니다.")
        else:
            for r in rows:
                source = r.get("source_name", "")
                title = r.get("title", "")
                url = r.get("final_url") or r.get("resolved_url") or r.get("google_news_url") or ""
                if url:
                    st.markdown(f"**[{source} - {title}]({url})**")
                else:
                    st.markdown(f"**{source} - {title}**")
                st.caption(
                    f"pubDate: {r.get('pubdate_ts') or r.get('pubdate_raw') or 'unknown'} | crawl: {r.get('crawl_status')}"
                )
                st.markdown("---")
    else:
        st.info("articles.jsonl이 아직 없습니다.")

    st.divider()

    # ---- 테마별 요약 & 큐레이션 ----
    theme_summaries_path = run_path / "theme_summaries.json"
    curations_path = run_path / "curations.json"

    theme_summaries = read_json(theme_summaries_path) if theme_summaries_path.exists() else {}
    curations = read_json(curations_path) if curations_path.exists() else {}

    st.subheader("테마별 요약 (10줄)")
    if not theme_summaries:
        st.info("아직 테마 요약이 없습니다.")
    else:
        for theme, obj in theme_summaries.items():
            st.markdown(f"### {theme}")
            lines = obj.get("lines", [])
            if lines:
                st.write("\n".join(lines))
            st.markdown("---")

    st.subheader("테마별 Top 15 큐레이션")
    if not curations:
        st.info("아직 큐레이션이 없습니다.")
    else:
        for theme, items in curations.items():
            st.markdown(f"### {theme}")

            if not items:
                st.caption("해당 테마에 큐레이션 결과가 없습니다.")
                st.markdown("---")
                continue

            for it in items[:15]:
                source = it.get("source_name", "")
                title = it.get("title", "")
                url = it.get("url", "")
                pub = it.get("pubdate_ts", "")
                score = it.get("score", "")

                if url:
                    st.markdown(f"**[{source} - {title}]({url})**")
                else:
                    st.markdown(f"**{source} - {title}**")

                meta = []
                if pub:
                    meta.append(f"🕒 {pub}")
                if score != "":
                    try:
                        meta.append(f"⭐ {float(score):.2f}")
                    except Exception:
                        meta.append(f"⭐ {score}")
                if meta:
                    st.caption(" | ".join(meta))

                summary_text = (it.get("summary_text") or "").strip()
                if summary_text:
                    st.write(summary_text)
                else:
                    s3 = it.get("summary_3_lines") or []
                    if s3:
                        st.write(" ".join([x.strip() for x in s3 if x and x.strip()]))
                st.markdown("---")

            st.divider()


# -----------------------------
# Page: History
# -----------------------------
elif page == "History":
    st.header("History: Run 목록 및 삭제")

    runs = list_runs()
    if not runs:
        st.info("아직 생성된 Run이 없습니다.")
        st.stop()

    for pth in runs:
        cols = st.columns([6, 2, 2])
        with cols[0]:
            st.write(pth.name)
        with cols[1]:
            try:
                run_meta = read_json(pth / "run.json")
                st.caption(run_meta.get("created_at", ""))
            except Exception:
                st.caption("run.json 읽기 실패")
        with cols[2]:
            if st.button("삭제", key=f"del_{pth.name}"):
                delete_run_folder(pth)
                st.success(f"삭제 완료: {pth.name}")
                st.rerun()


# -----------------------------
# Page: Settings
# -----------------------------
else:
    st.header("Settings: 매체/테마/키워드 설정")

    if not settings:
        st.error("settings_current.json을 찾지 못했습니다.")
        st.stop()

    st.caption("설정 변경은 다음 Run부터 반영됩니다.")

    tab1, tab2 = st.tabs(["매체", "테마/키워드"])

    with tab1:
        st.subheader("매체 목록")
        media = settings.get("media_sources", [])
        st.write(f"총 {len(media)}개")

        for i, m in enumerate(media):
            with st.expander(f"{m.get('name')} ({m.get('domain')})", expanded=False):
                m["enabled"] = st.checkbox("활성", value=bool(m.get("enabled", True)), key=f"media_en_{i}")
                m["group"] = st.selectbox(
                    "그룹",
                    options=["startup", "it", "econ", "daily"],
                    index=["startup", "it", "econ", "daily"].index(m.get("group", "econ"))
                    if m.get("group") in ["startup", "it", "econ", "daily"]
                    else 2,
                    key=f"media_group_{i}",
                )

        st.divider()
        st.subheader("매체 추가")
        new_domain = st.text_input("도메인 (예: example.com)")
        new_name = st.text_input("매체명 (예: 예시뉴스)")
        new_group = st.selectbox("그룹", ["startup", "it", "econ", "daily"], index=2)

        if st.button("매체 추가"):
            if new_domain and new_name:
                settings["media_sources"].append(
                    {"domain": new_domain.strip(), "name": new_name.strip(), "group": new_group, "enabled": True}
                )
                save_settings(settings)
                st.success("매체 추가 완료")
                st.rerun()
            else:
                st.warning("도메인과 매체명을 입력해주세요.")

    with tab2:
        st.subheader("테마 목록")
        themes = settings.get("themes", [])
        st.write(f"총 {len(themes)}개")

        for i, t in enumerate(themes):
            with st.expander(f"{t.get('name')}", expanded=False):
                t["enabled"] = st.checkbox("활성", value=bool(t.get("enabled", True)), key=f"theme_en_{i}")
                t["name"] = st.text_input("테마명", value=t.get("name", ""), key=f"theme_name_{i}")

                st.markdown("**Include Groups (AND/OR)**")
                st.caption("그룹끼리는 AND, 그룹 내부 키워드는 OR 입니다.")
                include_groups = t.get("include_groups", [[]])

                group_lines = []
                for g in include_groups:
                    group_lines.append(", ".join([x for x in g if x]))

                raw_groups = st.text_area(
                    "Include Groups (한 줄=한 그룹, 그룹 내 OR은 콤마)",
                    value="\n".join(group_lines),
                    height=120,
                    key=f"inc_groups_{i}",
                )
                parsed_groups = []
                for line in raw_groups.splitlines():
                    kws = [x.strip() for x in line.split(",") if x.strip()]
                    if kws:
                        parsed_groups.append(kws)
                t["include_groups"] = parsed_groups if parsed_groups else [[]]

                st.markdown("**Exclude Keywords (OR)**")
                ex_raw = st.text_input(
                    "Exclude (콤마로 구분)",
                    value=", ".join(t.get("exclude_keywords", [])),
                    key=f"exc_{i}",
                )
                t["exclude_keywords"] = [x.strip() for x in ex_raw.split(",") if x.strip()]

                # 필수 포함(AND): 아래 키워드 '모두' 포함되어야 테마 후보로 인정
                mi_raw = st.text_input(
                    "필수 포함 (AND, 콤마 구분) - 모두 포함되어야 함",
                    value=", ".join(t.get("must_include", [])),
                    key=f"must_inc_{i}",
                )
                t["must_include"] = [x.strip() for x in mi_raw.split(",") if x.strip()]

                # 필수 포함(OR): 아래 키워드 중 '하나라도' 포함되면 테마 후보로 인정
                mia_raw = st.text_input(
                    "필수 포함 (OR, 콤마 구분) - 하나라도 포함",
                    value=", ".join(t.get("must_include_any", [])),
                    key=f"must_any_{i}",
                )
                t["must_include_any"] = [x.strip() for x in mia_raw.split(",") if x.strip()]


                t["max_items"] = st.number_input(
                    "테마별 최대 큐레이션 수",
                    min_value=1,
                    max_value=30,
                    value=int(t.get("max_items", 15)),
                    key=f"max_{i}",
                )

                # (선택) must_include는 나중에 UI에 추가 가능
                if st.button("이 테마 삭제", key=f"del_theme_{i}"):
                    settings["themes"].pop(i)
                    save_settings(settings)
                    st.success("삭제 완료")
                    st.rerun()

        st.divider()
        st.subheader("테마 추가")
        new_theme_name = st.text_input("새 테마명")
        if st.button("테마 추가"):
            if new_theme_name.strip():
                settings["themes"].append(
                    {
                        "name": new_theme_name.strip(),
                        "enabled": True,
                        "include_groups": [[]],
                        "exclude_keywords": [],
                        "curation_priority": [],
                        "max_items": 15,
                    }
                )
                save_settings(settings)
                st.success("테마 추가 완료")
                st.rerun()
            else:
                st.warning("테마명을 입력해주세요.")

    st.divider()
    if st.button("설정 저장"):
        save_settings(settings)
        st.success("설정 저장 완료")
