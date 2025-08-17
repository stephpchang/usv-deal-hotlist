# hotlist_app.py
import os
from urllib.parse import urlparse, urlencode
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
DDLITE_URL = os.getenv("DDLITE_URL")  # e.g. http://localhost:8501 or your hosted Copilot

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
    s = series.astype(float)
    min_v = np.nanmin(s.values) if np.isfinite(s.values).any() else 0.0
    max_v = np.nanmax(s.values) if np.isfinite(s.values).any() else 0.0
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
    founder = 1.0 if fs in ("1", "true", "yes", "y") else 0.0
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
    founder = 1.0 if fs in ("1", "true", "yes", "y") else 0.0
    return {
        "recent_funding": round(funding * WEIGHTS["recent_funding"] * 100, 1),
        "growth_signal":  round(growth  * WEIGHTS["growth_signal"]  * 100, 1),
        "thematic_fit":   round(thematic* WEIGHTS["thematic_fit"]   * 100, 1),
        "founder_signal": round(founder * WEIGHTS["founder_signal"] * 100, 1),
    }

def deep_link_to_copilot(name: str, website: str) -> str | None:
    if not DDLITE_URL:
        return None
    q = {}
    if name: q["company"] = name
    if website: q["website"] = website
    return f"{DDLITE_URL}?{urlencode(q)}" if q else DDLITE_URL

# -----------------------------
# Demo dataset
# - 8 Seed/Series A demo entries
# - A few later-stage retained for optional comparison (when B+ is enabled)
# -----------------------------
data = [
    # --- Later-stage examples (only appear if you enable B+) ---
    dict(company="Hugging Face", thesis="AI / Open Source", stage="Series D / Growth",
         total_raised=235_000_000, last_round="Series D (2023)",
         notable_investors=["Sequoia", "Coatue", "Lux"],
         one_liner="Open-source hub for models, datasets, & tooling; the GitHub of AI.",
         website="https://huggingface.co/", sources=["https://huggingface.co/blog"],
         why_usv="Open networks & developer ecosystems; durable community moat.",
         why_now="Model hosting/inference partnerships expanding; developer pull remains strong.",
         intro_hint="Ask about paid usage mix, enterprise plans, and ecosystem monetization.",
         hiring_index=0.75, traffic_index=0.85, founder_signal=1),
    dict(company="Perplexity AI", thesis="AI / Machine Intelligence", stage="Series B / Growth",
         total_raised=165_000_000, last_round="Series B (2024)",
         notable_investors=["IVP", "NEA", "Jeff Bezos"],
         one_liner="AI-powered conversational search with cited answers across the live web.",
         website="https://www.perplexity.ai/", sources=["https://www.reuters.com/technology/"],
         why_usv="Aligns with 'open internet' & machine intelligence; strong engagement loops.",
         why_now="Exploding daily usage; moving from consumer curiosity to daily workflow.",
         intro_hint="Ask about retention cohorts and enterprise/education use cases.",
         hiring_index=0.9, traffic_index=0.95, founder_signal=1),

    # --- Seed / Series A (8 demo entries) ---
    dict(company="Oakleaf AI", thesis="AI / Machine Intelligence", stage="Seed",
         total_raised=3_200_000, last_round="Seed (2024)",
         notable_investors=["Demo Fund", "Angel Collective"],
         one_liner="Agents that triage L1 support tickets and hand off cleanly to humans.",
         website="https://example.com/oakleaf-ai", sources=["https://example.com/oakleaf-ai/press"],
         why_usv="Automation loop in daily workflows; measurable time-to-resolution impact.",
         why_now="Agent frameworks stabilizing; early design partners showing strong ROI.",
         intro_hint="Ask for pilot KPIs (deflections, CSAT) and security review status.",
         hiring_index=0.35, traffic_index=0.3, founder_signal=1),
    dict(company="Helio Grid", thesis="Climate Tech", stage="Seed",
         total_raised=5_000_000, last_round="Seed (2025)",
         notable_investors=["Earth Capital", "Seed Climate"],
         one_liner="API for distributed energy resource telemetry & dispatch for installers.",
         website="https://example.com/helio-grid", sources=["https://example.com/helio-grid/blog"],
         why_usv="Electrification infra; wedge via installers and utilities integrations.",
         why_now="DER incentives + hardware attach driving data interoperability demand.",
         intro_hint="Ask about utility integrations and paid pilots by region.",
         hiring_index=0.4, traffic_index=0.35, founder_signal=0),
    dict(company="Parcel Labs", thesis="Developer Tools", stage="Series A",
         total_raised=12_000_000, last_round="Series A (2025)",
         notable_investors=["CodeSeed", "Operator Ventures"],
         one_liner="Preview environments-as-a-service for every PR, with ephemeral databases.",
         website="https://example.com/parcellabs", sources=["https://example.com/parcellabs/changelog"],
         why_usv="Core dev loop; expands into test data mgmt and compliance snapshots.",
         why_now="Shift-left testing and platform teams standardizing on preview infra.",
         intro_hint="Ask for DORA metrics impact and enterprise POCs.",
         hiring_index=0.55, traffic_index=0.45, founder_signal=1),
    dict(company="GlacierID", thesis="Decentralized ID", stage="Seed",
         total_raised=4_800_000, last_round="Seed (2024)",
         notable_investors=["Open Identity Fund"],
         one_liner="Privacy-preserving identity assertions for apps, no PII exchange.",
         website="https://example.com/glacierid", sources=["https://example.com/glacierid/whitepaper"],
         why_usv="Open internet identity rails; unlocks spam resistance and fair drops.",
         why_now="Policy pressure + wallet adoption; developers want SDKs not protocols.",
         intro_hint="Ask for developer SDK adoption and validator distribution.",
         hiring_index=0.3, traffic_index=0.25, founder_signal=0),
    dict(company="Riverbed Finance", thesis="Fintech Infrastructure", stage="Series A",
         total_raised=10_000_000, last_round="Series A (2024)",
         notable_investors=["Rails Capital"],
         one_liner="Ledger & reconciliation APIs for marketplaces and B2B platforms.",
         website="https://example.com/riverbed", sources=["https://example.com/riverbed/docs"],
         why_usv="Financial backbones of platforms; expansion into risk & payments ops.",
         why_now="Marketplaces proliferating; CFO stacks consolidating around APIs.",
         intro_hint="Ask for gross volume under ledger and error rate improvements.",
         hiring_index=0.5, traffic_index=0.4, founder_signal=1),
    dict(company="OpenMesh", thesis="Open Data / Privacy Infra", stage="Seed",
         total_raised=2_500_000, last_round="Seed (2025)",
         notable_investors=["Network Angels"],
         one_liner="Peer-to-peer data lake with verifiable lineage and access controls.",
         website="https://example.com/openmesh", sources=["https://example.com/openmesh/roadmap"],
         why_usv="User-owned data primitives; composable building blocks for apps.",
         why_now="AI data provenance demands verifiable pipelines; open infra emerging.",
         intro_hint="Ask for early ecosystem apps and data provider incentives.",
         hiring_index=0.28, traffic_index=0.22, founder_signal=0),
    dict(company="Commons Health OS", thesis="Healthcare", stage="Series A",
         total_raised=9_000_000, last_round="Series A (2025)",
         notable_investors=["Health Seed", "City Angels"],
         one_liner="Care navigation co-pilot for FQHCs; integrates scheduling & referrals.",
         website="https://example.com/commons-health", sources=["https://example.com/commons-health/case-study"],
         why_usv="Civic networks + software; expands to claims and community orgs.",
         why_now="FQHCs digitizing; reimbursement rules encourage navigation tooling.",
         intro_hint="Ask for signed LOIs, rollout speed, and payer integrations.",
         hiring_index=0.4, traffic_index=0.3, founder_signal=1),
    dict(company="Nimbus Agent", thesis="AI / Open Source", stage="Seed",
         total_raised=3_800_000, last_round="Seed (2025)",
         notable_investors=["OSS Capital (demo)"],
         one_liner="Open-source agent runtime with guardrails and skills marketplace.",
         website="https://example.com/nimbus-agent", sources=["https://example.com/nimbus-agent/github"],
         why_usv="Developer ecosystems & open networks; compounding via community.",
         why_now="Agent stacks consolidating; need safe, modular runtimes.",
         intro_hint="Ask for GitHub stars-to-active conversions and paid support pipeline.",
         hiring_index=0.45, traffic_index=0.33, founder_signal=1),

    # --- Original early entries kept ---
    dict(company="Teller", thesis="Fintech Infrastructure", stage="Seed / Early",
         total_raised=6_000_000, last_round="Seed (2022)",
         notable_investors=["Lightspeed", "SciFi VC"],
         one_liner="API platform for secure access to U.S. bank data—no screen scraping.",
         website="https://www.teller.io/", sources=["https://teller.io/blog"],
         why_usv="Open finance rails; cleaner data access for consumer/SMB fintech.",
         why_now="Banks tightening; developers want direct, reliable data integrations.",
         intro_hint="Ask about bank coverage, latency/reliability SLOs, and top fintech adopters.",
         hiring_index=0.25, traffic_index=0.3, founder_signal=0),
    dict(company="Plaid Climate", thesis="Climate + Fintech", stage="Spin-out / Early",
         total_raised=None, last_round="—", notable_investors=["Plaid (parent)"],
         one_liner="Sustainability tools layered on transaction data for climate impact tracking.",
         website="https://plaid.com/", sources=["https://plaid.com/blog/"],
         why_usv="Intersection of open finance and climate; unique data signal for impact reporting.",
         why_now="Banks/fintechs under pressure to offer climate reporting to consumers/SMBs.",
         intro_hint="Ask about pilot partners and accuracy of emissions estimation models.",
         hiring_index=0.3, traffic_index=0.4, founder_signal=0),
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
# De-duplication
# -----------------------------
dedupe_before = len(df)
df = df.sort_values(["company"]).drop_duplicates(subset=["domain"], keep="first")
df = df.drop_duplicates(subset=["company"], keep="first")
deduped_count = dedupe_before - len(df)

# -----------------------------
# Header
# -----------------------------
st.title("🔥 USV Deal Hotlist")
st.subheader("Curated companies aligned with USV’s theses.")
st.caption("Demo uses sample entries and curated notes. Replace with real deals as needed.")
if deduped_count > 0:
    st.caption(f"De-duplicated {deduped_count} item(s) by domain/company.")

# -----------------------------
# Session state for assignments & status
# -----------------------------
if "owners" not in st.session_state: st.session_state["owners"] = {}
if "status" not in st.session_state: st.session_state["status"] = {}
if "notes_map" not in st.session_state: st.session_state["notes_map"] = {}
STATUS_OPTS = ["New", "Reviewing", "Waiting on Data", "Pass", "Advance"]

# -----------------------------
# Filters (Seed & A default + Guarantee 5)
# -----------------------------
with st.sidebar:
    st.header("Filter")
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

# Apply filters
f = base.copy()
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

# Apply min score to the main set
f = f[f["score"] >= min_score]

# ---------- Guarantee at least 5 cards (respecting early-stage setting) ----------
view = f.copy()
if guarantee_min and len(view) < 5:
    pool = df.copy()
    if not include_late:
        pool = pool[pool["stage_bucket"].isin(["Seed", "Series A"])]
    # Respect current filters for relevance (except min_score so we can actually fill to 5)
    if pick_thesis != "All":
        pool = pool[pool["thesis"] == pick_thesis]
    if pick_stage != "All":
        pool = pool[pool["stage"] == pick_stage]
    pool = pool[pool["total_raised"].fillna(0) >= (min_amt * 1_000_000)]
    if hide_pass:
        pass_names = {n for n, s in st.session_state["status"].items() if s == "Pass"}
        pool = pool[~pool["company"].isin(pass_names)]
    # Score & order the pool
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
            st.markd
