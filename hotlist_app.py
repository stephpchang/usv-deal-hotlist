# hotlist_app.py
import os
from urllib.parse import urlencode, urlparse
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from io import BytesIO
from importlib.util import find_spec

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="USV Deal Hotlist", layout="wide")

# -----------------------------
# Config: scoring & optional deep link to DD Copilot
# -----------------------------
WEIGHTS = {
    "recent_funding": 0.35,   # uses recent_funding_usd (or total_raised) normalized 0..1
    "growth_signal": 0.25,    # avg of hiring_index & traffic_index (0..1)
    "thematic_fit": 0.25,     # 1 if thesis in focus_theses, else 0
    "founder_signal": 0.15,   # 1 if founder_signal truthy, else 0
}
THESES = [
    "AI / Machine Intelligence", "AI / Open Source", "Climate Tech", "Climate + Fintech",
    "Developer Tools", "Fintech Infrastructure", "Open Data / Privacy Infra",
    "Decentralized ID", "Open Internet / DeFi", "Healthcare"
]
DDLITE_URL = os.getenv("DDLITE_URL")  # e.g. https://your-dd-copilot.app

# -----------------------------
# Helpers
# -----------------------------
def tznow():
    return datetime.now(timezone.utc)

def fmt_money(n: int | float | None) -> str:
    if n is None or pd.isna(n):
        return "—"
    n = float(n)
    if n >= 1_000_000_000:
        return f"${n/1_000_000_000:.1f}B"
    return f"${n/1_000_000:.1f}M"

def pick_excel_engine() -> str | None:
    if find_spec("xlsxwriter"):
        return "xlsxwriter"
    if find_spec("openpyxl"):
        return "openpyxl"
    return None

EXCEL_ENGINE = pick_excel_engine()

def as_badge(txt: str) -> str:
    safe = str(txt)
    return (
        "<span style='background:#eef2ff;border:1px solid #c7d2fe;"
        "border-radius:999px;padding:2px 8px;font-size:12px;color:#3730a3'>"
        f"{safe}</span>"
    )

def canonical_domain(url_or_domain: str) -> str:
    if not url_or_domain:
        return ""
    s = str(url_or_domain).strip()
    if "://" not in s:
        s = "https://" + s
    try:
        host = urlparse(s).netloc.lower().split(":")[0]
        return host
    except Exception:
        return str(url_or_domain).lower()

def normalize_series_0_1(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float).fillna(0.0)
    vals = s.values
    finite_any = np.isfinite(vals).any()
    if not finite_any:
        return pd.Series([0.0] * len(s), index=s.index)
    min_v = float(np.nanmin(vals))
    max_v = float(np.nanmax(vals))
    if max_v <= min_v:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - min_v) / (max_v - min_v)

def stage_bucket(stage_text: str) -> str:
    """Map raw stage text to simple buckets for filtering."""
    s = (stage_text or "").lower().replace("-", " ")
    if "pre seed" in s or "preseed" in s:
        return "Seed"
    if "series a" in s:
        return "Series A"
    if "seed" in s:
        return "Seed"
    return "Later (B+)"

def stage_order(bucket: str) -> int:
    return {"Seed": 0, "Series A": 1, "Later (B+)": 2}.get(bucket, 9)

def compute_score_row(row: pd.Series, focus_theses: list[str]) -> float:
    funding = float(row.get("recent_funding_usd_norm") or 0.0)  # 0..1
    h = row.get("hiring_index", np.nan)
    t = row.get("traffic_index", np.nan)
    growth = float(np.nanmean([h, t]))
    if np.isnan(growth):
        growth = 0.0
    growth = max(0.0, min(1.0, growth))
    thematic = 1.0 if focus_theses and (row.get("thesis") in focus_theses) else 0.0
    fs = str(row.get("founder_signal", "")).strip().lower()
    founder = 1.0 if fs in ("1", "true", "yes", "y") or row.get("founder_signal") == 1 else 0.0
    score = (
        funding * WEIGHTS["recent_funding"] +
        growth  * WEIGHTS["growth_signal"] +
        thematic* WEIGHTS["thematic_fit"] +
        founder * WEIGHTS["founder_signal"]
    )
    return round(score * 100.0, 1)

def score_breakdown(row: pd.Series, focus_theses: list[str]) -> dict:
    funding = float(row.get("recent_funding_usd_norm") or 0.0)
    h = row.get("hiring_index", np.nan)
    t = row.get("traffic_index", np.nan)
    growth = float(np.nanmean([h, t]))
    if np.isnan(growth):
        growth = 0.0
    growth = max(0.0, min(1.0, growth))
    thematic = 1.0 if focus_theses and (row.get("thesis") in focus_theses) else 0.0
    fs = str(row.get("founder_signal", "")).strip().lower()
    founder = 1.0 if fs in ("1", "true", "yes", "y") or row.get("founder_signal") == 1 else 0.0
    return {
        "recent_funding": round(funding * WEIGHTS["recent_funding"] * 100, 1),
        "growth_signal":  round(growth  * WEIGHTS["growth_signal"]  * 100, 1),
        "thematic_fit":   round(thematic* WEIGHTS["thematic_fit"]   * 100, 1),
        "founder_signal": round(founder * WEIGHTS["founder_signal"] * 100, 1),
    }

def deep_link_to_copilot(name: str, website: str) -> str | None:
    base = DDLITE_URL
    if not base:
        return None
    q = {}
    if name: q["company"] = name
    if website: q["website"] = website
    return f"{base}?{urlencode(q)}" if q else base

# -----------------------------
# Built-in dataset: 5 REAL Seed/Series A companies
# -----------------------------
data = [
    dict(
        company="Together AI",
        thesis="AI / Machine Intelligence",
        stage="Series A",
        total_raised=106_000_000,
        last_round="Series A (Mar 2024)",
        notable_investors=["Salesforce Ventures", "Coatue", "Lux Capital", "Kleiner Perkins", "Emergence"],
        one_liner="Cloud platform to train, fine-tune, and serve open/custom AI models.",
        website="https://www.together.ai/",
        sources=[
            "https://www.together.ai/blog/series-a2",
            "https://www.reuters.com/technology/together-ai-valued-125-bln-salesforce-led-funding-round-2024-03-13/"
        ],
        why_usv="Aligns with USV’s ‘open internet + machine intelligence’ thesis; developer-first infra with strong ecosystem pull.",
        why_now="Rapid enterprise shift to open/model-mix stacks; need for lower-cost, flexible training/serving.",
        intro_hint="Ask about enterprise GPU capacity SLAs, fine-tune economics vs proprietary clouds, and top customer references.",
        hiring_index=0.55, traffic_index=0.60, founder_signal=1
    ),
    dict(
        company="Modal Labs",
        thesis="Developer Tools",
        stage="Series A",
        total_raised=23_000_000,
        last_round="Series A (Oct 2023)",
        notable_investors=["Redpoint", "Amplify Partners", "Lux Capital", "Definition Capital"],
        one_liner="Serverless compute to run Python/ML/AI workloads in the cloud without managing infra.",
        website="https://modal.com/",
        sources=["https://modal.com/blog/general-availability-and-series-a-press-release"],
        why_usv="Core developer workflow + infra leverage; compounding usage as apps scale.",
        why_now="AI/ETL jobs exploding; teams want on-demand GPUs and batch at sane DX and costs.",
        intro_hint="Ask for customer case studies (GPU hours, cost deltas), SOC2 posture, and VPC/private networking roadmap.",
        hiring_index=0.45, traffic_index=0.50, founder_signal=1
    ),
    dict(
        company="Quilt",
        thesis="Climate Tech",
        stage="Series A",
        total_raised=42_000_000,  # $9M seed + $33M Series A
        last_round="Series A (Apr 2024)",
        notable_investors=["Energy Impact Partners", "Galvanize Climate Solutions", "Lowercarbon", "Gradient Ventures"],
        one_liner="Smart home heat pump system—hardware + app + installer network for electrified HVAC.",
        website="https://www.quilt.com/",
        sources=[
            "https://www.quilt.com/news/quilt-raises-33m-to-launch-the-smartest-way-to-heat-and-cool-your-home",
            "https://techcrunch.com/2024/04/16/quilt-heat-pump-series-a/"
        ],
        why_usv="Electrification wedge with software touchpoints; potential network effects across install base and grid services.",
        why_now="Heat-pump adoption surging with policy incentives; consumer UX differentiation matters.",
        intro_hint="Ask about install throughput, CAC by channel, and grid/utility partnership pipeline.",
        hiring_index=0.40, traffic_index=0.35, founder_signal=0
    ),
    dict(
        company="Hippocratic AI",
        thesis="Healthcare",
        stage="Series A",
        total_raised=120_000_000,
        last_round="Series A (Mar 2024)",
        notable_investors=["General Catalyst", "Premji Invest", "a16z", "SV Angel", "NVIDIA (strategic)"],
        one_liner="Safety-focused healthcare AI agents for staffing and patient engagement (non-diagnostic).",
        website="https://www.hippocraticai.com/",
        sources=[
            "https://www.globenewswire.com/news-release/2024/03/18/2848237/0/en/Hippocratic-AI-Raises-53-Million-Series-A-at-a-500-Million-Valuation.html",
            "https://www.fiercehealthcare.com/ai-and-machine-learning/hippocratic-ai-banks-53m-backed-general-catalyst-a16z-memorial-hermann-uhs"
        ],
        why_usv="Large vertical where safe AI agents can be defensible; strong health-system participation.",
        why_now="Acute labor shortages and call-center backlog; payers/providers piloting agent workflows.",
        intro_hint="Ask for live deployments by service line, clinical safety guardrails, and phase-three testing plan.",
        hiring_index=0.50, traffic_index=0.45, founder_signal=1
    ),
    dict(
        company="Privy",
        thesis="Open Internet / DeFi",
        stage="Series A",
        total_raised=26_300_000,  # ~$18M A + ~$8.3M seed
        last_round="Series A (Nov 2023)",
        notable_investors=["Paradigm", "Sequoia", "BlueYard", "Archetype"],
        one_liner="SDK/APIs to onboard mainstream users on-chain with embedded, self-custodial wallets.",
        website="https://www.privy.io/",
        sources=[
            "https://www.privy.io/blog/series-a-announcement",
            "https://www.finsmes.com/2023/11/privy-raises-18m-in-series-a-funding.html",
            "https://www.theblock.co/post/347110/privy-raises-15-million-usd-round-led-by-ribbit-capital-investment-in-web3-wallet-infrastructure-heats-up"
        ],
        why_usv="Identity + wallet infra for the open internet; reduces onboarding friction for consumer crypto apps.",
        why_now="Wallet UX is the bottleneck; apps want embedded wallets and familiar auth flows.",
        intro_hint="Ask for conversion metrics (signup→active), largest production apps, and compliance posture.",
        hiring_index=0.45, traffic_index=0.40, founder_signal=0
    ),
]

df = pd.DataFrame(data)

# -----------------------------
# Derived fields: domain, funding normalization, stage bucket
# -----------------------------
df["domain"] = df["website"].apply(canonical_domain)
df["recent_funding_usd"] = df.get("recent_funding_usd", pd.Series([np.nan]*len(df)))
df["recent_funding_usd"] = df["recent_funding_usd"].fillna(df["total_raised"])
df["recent_funding_usd_norm"] = normalize_series_0_1(df["recent_funding_usd"].fillna(0))
df["stage_bucket"] = df["stage"].apply(stage_bucket)

# -----------------------------
# De-duplication (company ONLY)
# -----------------------------
before = len(df)
df = df.sort_values(["company"]).drop_duplicates(subset=["company"], keep="first")
deduped_count = before - len(df)

# -----------------------------
# Header
# -----------------------------
st.title("🔥 USV Deal Hotlist")
st.subheader("Seed & Series A–first, score-ranked shortlist for USV.")
st.markdown(
    "> **What this is:** A prioritized pipeline for quick sourcing and partner review.\n"
    "> Assign owners, set status, and jump to DD Copilot for structured diligence."
)
st.caption("Dataset uses public information; replace or extend with your own list anytime.")
if deduped_count > 0:
    st.caption(f"De-duplicated {deduped_count} by company.")

# -----------------------------
# Session state for assignments & status
# -----------------------------
if "owners" not in st.session_state: st.session_state["owners"] = {}
if "status" not in st.session_state: st.session_state["status"] = {}
if "notes_map" not in st.session_state: st.session_state["notes_map"] = {}
STATUS_OPTS = ["New", "Reviewing", "Waiting on Data", "Pass", "Advance"]

# -----------------------------
# Filters (Seed & A default + Mode + Guarantee 5)
# -----------------------------
with st.sidebar:
    st.header("Filter")

    mode = st.radio("Mode", ["Sourcing", "Partner meeting"], index=0, horizontal=True)

    # Reset sticky widget state, if needed
    if st.button("Reset filters"):
        st.session_state.clear()
        st.experimental_rerun()

    include_late = st.checkbox("Include later-stage (Series B+)", value=False)
    guarantee_min = st.checkbox("Guarantee at least 5 cards (auto-expand)", value=True)

    base = df.copy()
    if not include_late:
        base = base[base["stage_bucket"].isin(["Seed", "Series A"])]

    thesis_opts = ["All"] + sorted(base["thesis"].unique())
    stage_opts = ["All"] + sorted(base["stage"].unique())
    pick_thesis = st.selectbox("Thesis", thesis_opts, index=0)
    pick_stage = st.selectbox("Stage", stage_opts, index=0)
    min_amt = st.slider("Min total raised ($M)", 0, int((base["total_raised"].fillna(0).max()) / 1_000_000), 0)

    st.markdown("---")
    focus_theses = st.multiselect("Focus theses (score boost)", THESES, default=[])
    min_score = st.slider("Min score", 0, 100, 0)
    sort_opts = [
        "Stage (Seed→A→B+) then score",
        "Highest score",
        "Largest total raised",
        "Company A→Z",
        "Most recent round label",
    ]
    sort_by = st.selectbox("Sort by", sort_opts, index=0)

    st.markdown("---")
    hide_pass = st.checkbox("Hide 'Pass' status", value=False)

# Partner meeting intent: Seed/A only, tight top 5
meeting_mode = (mode == "Partner meeting")
if meeting_mode:
    include_late = False
    guarantee_min = True

# Apply filters
f = df.copy()
if not include_late:
    f = f[f["stage_bucket"].isin(["Seed", "Series A"])]
if pick_thesis != "All":
    f = f[f["thesis"] == pick_thesis]
if pick_stage != "All":
    f = f[f["stage"] == pick_stage]
f = f[f["total_raised"].fillna(0) >= (min_amt * 1_000_000)]

# Compute scores
f["score"] = f.apply(lambda r: compute_score_row(r, focus_theses), axis=1)
f["_breakdown"] = f.apply(lambda r: score_breakdown(r, focus_theses), axis=1)

# Optional: hide 'Pass'
def row_status(name): return st.session_state["status"].get(name, "New")
if hide_pass:
    mask = f["company"].apply(lambda n: row_status(n) != "Pass")
    f = f[mask]

# Apply min score
f = f[f["score"] >= min_score]

# ---------- Guarantee at least 5 cards (respect early-stage + filters) ----------
view = f.copy()
if guarantee_min and len(view) < 5:
    pool = df.copy()
    if not include_late:
        pool = pool[pool["stage_bucket"].isin(["Seed", "Series A"])]
    if pick_thesis != "All":
        pool = pool[pool["thesis"] == pick_thesis]
    if pick_stage != "All":
        pool = pool[pool["stage"] == pick_stage]
    pool = pool[pool["total_raised"].fillna(0) >= (min_amt * 1_000_000)]
    if hide_pass:
        pass_names = {n for n, s in st.session_state["status"].items() if s == "Pass"}
        pool = pool[~pool["company"].isin(pass_names)]

    pool["score"] = pool.apply(lambda r: compute_score_row(r, focus_theses), axis=1)
    pool["_breakdown"] = pool.apply(lambda r: score_breakdown(r, focus_theses), axis=1)
    pool["_stage_order"] = pool["stage_bucket"].apply(stage_order)
    pool = pool[~pool["company"].isin(set(view["company"]))]
    pool = pool.sort_values(["_stage_order", "score"], ascending=[True, False], na_position="last")

    extras_needed = 5 - len(view)
    extras = pool.head(extras_needed).copy()
    extras["_showcase_extra"] = True
    view = pd.concat([view, extras], ignore_index=True)

# -----------------------------
# Sorting for display
# -----------------------------
view["_stage_order"] = view["stage_bucket"].apply(stage_order)

# If later-stage is excluded, force stage-first ordering regardless of dropdown
if not include_late:
    view = view.sort_values(["_stage_order", "score"], ascending=[True, False], na_position="last")
else:
    if sort_by == "Largest total raised":
        view = view.sort_values("total_raised", ascending=False, na_position="last")
    elif sort_by == "Company A→Z":
        view = view.sort_values("company", ascending=True, na_position="last")
    elif sort_by == "Most recent round label":
        view = view.sort_values("last_round", ascending=False, na_position="last")
    elif sort_by == "Stage (Seed→A→B+) then score":
        view = view.sort_values(["_stage_order", "score"], ascending=[True, False], na_position="last")
    else:  # Highest score
        view = view.sort_values("score", ascending=False, na_position="last")

# Meeting mode: compact top 5
if meeting_mode:
    view = view.head(5)

# -----------------------------
# Summary metrics
# -----------------------------
total_companies = len(view)
total_raised = view["total_raised"].sum(skipna=True)
avg_raised = view["total_raised"].mean(skipna=True) if total_companies > 0 else None
avg_score = round(float(view["score"].mean()), 1) if total_companies > 0 else None

def _flatten(xs):
    out = []
    for row in xs.dropna():
        out.extend(row)
    return out

unique_investors = sorted(set(_flatten(view["notable_investors"])) - {""}) if total_companies else []
st.markdown("### Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Companies", total_companies)
c2.metric("Total raised", fmt_money(total_raised))
c3.metric("Avg total raised", fmt_money(avg_raised))
c4.metric("Notable investors", len(unique_investors))
c5.metric("Avg score", "—" if avg_score is None else f"{avg_score}")

# -----------------------------
# Results
# -----------------------------
st.markdown("### Companies")
if total_companies == 0:
    st.info("No companies match your filters.")
else:
    for _, r in view.iterrows():
        name = r["company"]
        with st.container(border=True):
            # Title + badges
            badges = [
                f"**{name}**",
                as_badge(r["thesis"]),
                as_badge(r["stage"]),
                as_badge(fmt_money(r["total_raised"])),
                as_badge(f"Score {r['score']}")
            ]
            if r.get("_showcase_extra"):
                badges.append(as_badge("Showcase extra"))
            st.markdown("  ".join(badges), unsafe_allow_html=True)

            # One-liner
            st.write(r["one_liner"])

            # Key facts
            c1, c2, c3, c4 = st.columns([3, 3, 3, 3])
            c1.write(f"**Last round:** {r['last_round']}")
            c2.write(f"**Website:** [{r['website']}]({r['website']})")
            invs = ", ".join(r["notable_investors"]) if isinstance(r["notable_investors"], list) else "—"
            c3.write(f"**Notable investors:** {invs if invs else '—'}")
            c4.write(f"**Thesis:** {r['thesis']}")

            # Score breakdown
            with st.expander("Score breakdown"):
                bd = r["_breakdown"]
                st.write(
                    f"- Recent funding: **{bd['recent_funding']}**"
                    f"\n- Growth signal: **{bd['growth_signal']}**"
                    f"\n- Thematic fit: **{bd['thematic_fit']}**"
                    f"\n- Founder signal: **{bd['founder_signal']}**"
                    f"\n\n**Total:** {r['score']}"
                )

            # Why USV / Why now
            st.write(f"**Why USV:** {r['why_usv']}")
            st.write(f"**Why now:** {r['why_now']}")

            # Sources
            if isinstance(r["sources"], list) and r["sources"]:
                links = " · ".join([f"[source]({u})" for u in r["sources"]])
                st.write(f"**Sources:** {links}")

            # Assignment + Status + Deep link
            if meeting_mode:
                link = deep_link_to_copilot(name, r["website"])
                if link:
                    st.link_button("Open in DD Copilot", link, use_container_width=False)
                else:
                    st.caption("Set DDLITE_URL to enable Copilot deep link")
            else:
                a1, a2, a3, a4 = st.columns([2, 2, 3, 3])

                owner_val = st.session_state["owners"].get(name, "")
                st.session_state["owners"][name] = a1.text_input("Owner", value=owner_val, key=f"owner_{name}").strip()

                status_val = st.session_state["status"].get(name, "New")
                st.session_state["status"][name] = a2.selectbox(
                    "Status", STATUS_OPTS,
                    index=STATUS_OPTS.index(status_val) if status_val in STATUS_OPTS else 0,
                    key=f"status_{name}"
                )

                note_val = st.session_state["notes_map"].get(name, "")
                st.session_state["notes_map"][name] = a3.text_input("Short note", value=note_val, key=f"note_{name}")

                link = deep_link_to_copilot(name, r["website"])
                if link:
                    a4.link_button("Open in DD Copilot", link, use_container_width=True)
                else:
                    a4.caption("Set DDLITE_URL to enable Copilot deep link")

            # Next action
            ask = r.get("intro_hint") or "Ask for 3 customer references and latest traction metrics."
            outreach = f"Hi — exploring {name} for USV’s thesis. Could you intro me to the team? {ask}"
            st.write("**Next action:**")
            st.code(outreach, language="text")

# -----------------------------
# Export (partner packet)
# -----------------------------
st.subheader("Export (partner packet)")
export_cols = [
    "company", "thesis", "stage", "total_raised", "last_round",
    "notable_investors", "one_liner", "website", "sources", "why_usv", "why_now",
    "score"
]
exp = view[export_cols].copy()
exp["owner"] = exp["company"].apply(lambda n: st.session_state["owners"].get(n, ""))
exp["status"] = exp["company"].apply(lambda n: st.session_state["status"].get(n, "New"))
exp["note"]   = exp["company"].apply(lambda n: st.session_state["notes_map"].get(n, ""))

csv = exp.to_csv(index=False)
st.download_button("Download CSV", csv, "usv_deal_hotlist.csv", "text/csv", use_container_width=True)

if EXCEL_ENGINE:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine=EXCEL_ENGINE) as writer:
        exp.to_excel(writer, index=False, sheet_name="Hotlist")
    st.download_button(
        f"Download Excel ({EXCEL_ENGINE})",
        buf.getvalue(),
        "usv_deal_hotlist.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
else:
    st.info("Excel export unavailable: add `XlsxWriter` or `openpyxl` to requirements.txt. CSV export works above.")
