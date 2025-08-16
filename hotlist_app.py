# hotlist_app.py — Early-Stage Hotlist (USV Core)
import os
from urllib.parse import urlparse, urlencode
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from io import BytesIO
from importlib.util import find_spec

# ---------------------------------
# Page setup
# ---------------------------------
st.set_page_config(page_title="Early-Stage Hotlist (USV Core)", layout="wide")

# ---------------------------------
# Config
# ---------------------------------
DDLITE_URL = os.getenv("DDLITE_URL")  # Optional deep link target

ALL_THESES = [
    "AI / Machine Intelligence", "AI / Open Source",
    "Developer Tools", "Fintech Infrastructure",
    "Open Data / Privacy Infra",
    # (We keep Climate/Crypto in enum but exclude by default)
    "Climate Tech", "Climate + Fintech", "Open Internet / DeFi", "Decentralized ID",
]

# By default we EXCLUDE these from the Core hotlist (toggle in UI if you want)
EXCLUDED_THESES_DEFAULT = {
    "Climate Tech", "Climate + Fintech", "Open Internet / DeFi", "Decentralized ID"
}

EARLY_STAGE_KEYS = {"pre-seed", "seed", "series a"}

STAGE_WEIGHTS_CORE = {
    "pre-seed": 1.00, "seed": 1.00, "series a": 0.90,
    "series b": 0.55, "series c": 0.35, "series d": 0.25,
    "growth": 0.20, "spin-out": 0.45, "unknown": 0.40,
}

WEIGHTS_CORE = {  # Seed/A bias + founder-forward
    "stage_fit":      0.35,
    "founder_signal": 0.30,
    "thematic_fit":   0.25,
    "growth_signal":  0.05,
    "capital_eff":    0.05,
}

WEIGHTS_NEUTRAL = {  # legacy-style fallback
    "recent_funding": 0.35,
    "growth_signal":  0.25,
    "thematic_fit":   0.25,
    "founder_signal": 0.15,
}

# ---------------------------------
# Helpers
# ---------------------------------
def fmt_money(n: int | float | None) -> str:
    if n is None or pd.isna(n):
        return "—"
    n = float(n)
    if n >= 1_000_000_000:
        return f"${n/1_000_000_000:.1f}B"
    return f"${n/1_000_000:.1f}M"

def pick_excel_engine() -> str | None:
    if find_spec("xlsxwriter"): return "xlsxwriter"
    if find_spec("openpyxl"):   return "openpyxl"
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
    if not url_or_domain: return ""
    s = str(url_or_domain).strip()
    if "://" not in s: s = "https://" + s
    try:
        return urlparse(s).netloc.lower().split(":")[0]
    except Exception:
        return str(url_or_domain).lower()

def canonical_stage(label: str) -> str:
    if not label: return "unknown"
    x = label.lower()
    if "pre-seed" in x or "pre seed" in x: return "pre-seed"
    if "seed" in x: return "seed"
    if "series a" in x: return "series a"
    if "series b" in x: return "series b"
    if "series c" in x: return "series c"
    if "series d" in x: return "series d"
    if "growth" in x:   return "growth"
    if "spin" in x:     return "spin-out"
    return "unknown"

def normalize_series_0_1(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    vals = s.values.astype(float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    min_v = float(np.nanmin(vals)); max_v = float(np.nanmax(vals))
    if max_v <= min_v:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - min_v) / (max_v - min_v)

def deep_link_to_copilot(name: str, website: str) -> str | None:
    base = DDLITE_URL
    if not base: return None
    q = {}
    if name: q["company"] = name
    if website: q["website"] = website
    return f"{base}?{urlencode(q)}" if q else base

# ---------------------------------
# Demo dataset (expanded with Seed/A Core-aligned examples)
# NOTE: Amounts/rounds are placeholders for demo usability.
# ---------------------------------
data = [
    # --- Seed / Series A (Core-aligned) — new entries ---
    dict(company="Cursor", thesis="Developer Tools", stage="Series A",
         total_raised=14_000_000, last_round="Series A (2024)",
         notable_investors=["Placeholder"], one_liner="AI-native code editor to accelerate software development.",
         website="https://www.cursor.com/", sources=["https://www.cursor.com/"], why_usv="Daily dev workflow + AI loops.",
         why_now="AI pair programming adoption accelerating; strong bottoms-up motion.",
         intro_hint="Ask about weekly active devs, projects per user, and team seats.",
         hiring_index=0.6, traffic_index=0.7, founder_signal=1),

    dict(company="Continue.dev", thesis="Developer Tools", stage="Seed",
         total_raised=3_000_000, last_round="Seed (2024)",
         notable_investors=["Placeholder"], one_liner="Open-source local-first AI coding assistant for IDEs.",
         website="https://www.continue.dev/", sources=["https://www.continue.dev/"], why_usv="Open-source + developer network effects.",
         why_now="Developers want private/local AI assist and extensible tools.",
         intro_hint="Ask about plugin ecosystem momentum and retention cohorts.",
         hiring_index=0.4, traffic_index=0.45, founder_signal=1),

    dict(company="Modal Labs", thesis="Developer Tools", stage="Series A",
         total_raised=16_000_000, last_round="Series A (2023)",
         notable_investors=["Placeholder"], one_liner="Serverless compute designed for AI and data workloads.",
         website="https://modal.com/", sources=["https://modal.com/blog"], why_usv="Infra wedge with strong dev love; compounding workflows.",
         why_now="AI jobs need elastic GPU/CPU with simple DX; migration from DIY infra.",
         intro_hint="Ask about job volume growth and enterprise conversion.",
         hiring_index=0.55, traffic_index=0.6, founder_signal=1),

    dict(company="Dust", thesis="AI / Machine Intelligence", stage="Seed",
         total_raised=5_000_000, last_round="Seed (2023)",
         notable_investors=["Placeholder"], one_liner="Workspace agents that connect to your tools and data.",
         website="https://www.dust.tt/", sources=["https://www.dust.tt/blog"], why_usv="Agents + integrations → compounding utility.",
         why_now="Teams standardizing on AI workflows beyond chat; measurable productivity.",
         intro_hint="Ask about active workflows per org and integration depth.",
         hiring_index=0.35, traffic_index=0.4, founder_signal=0),

    dict(company="LlamaIndex", thesis="AI / Open Source", stage="Seed",
         total_raised=8_000_000, last_round="Seed (2023)",
         notable_investors=["Placeholder"], one_liner="Open-source framework for building RAG/data apps on LLMs.",
         website="https://www.llamaindex.ai/", sources=["https://www.llamaindex.ai/blog"], why_usv="Open ecosystem + developer pull.",
         why_now="Data-native AI apps spreading; need standard tooling for retrieval.",
         intro_hint="Ask about paid usage mix vs OSS usage and enterprise plans.",
         hiring_index=0.45, traffic_index=0.55, founder_signal=1),

    dict(company="OpenPipe", thesis="AI / Machine Intelligence", stage="Seed",
         total_raised=4_000_000, last_round="Seed (2024)",
         notable_investors=["Placeholder"], one_liner="Model fine-tuning and eval platform to ship production LLMs.",
         website="https://www.openpipe.ai/", sources=["https://www.openpipe.ai/"], why_usv="Quality/economics lever for AI apps.",
         why_now="Teams seeking lower cost & higher accuracy via fine-tunes.",
         intro_hint="Ask about eval wins and cost savings vs base models.",
         hiring_index=0.35, traffic_index=0.45, founder_signal=0),

    dict(company="Convex", thesis="Developer Tools", stage="Series A",
         total_raised=26_000_000, last_round="Series A (2023)",
         notable_investors=["Placeholder"], one_liner="Stateful backend for modern web apps—no-ops data & auth.",
         website="https://www.convex.dev/", sources=["https://stack.convex.dev/"], why_usv="DX wedge; simplifies app data + realtime.",
         why_now="Fullstack teams want speed; infra abstraction winning.",
         intro_hint="Ask about production app counts and team expansion.",
         hiring_index=0.5, traffic_index=0.55, founder_signal=0),

    dict(company="Baseten", thesis="Developer Tools", stage="Series A",
         total_raised=20_000_000, last_round="Series A (2022)",
         notable_investors=["Placeholder"], one_liner="Inference infra and tooling to serve and observe ML models.",
         website="https://www.baseten.co/", sources=["https://www.baseten.co/blog"], why_usv="Model → prod gap; sticky infra.",
         why_now="Explosion of model deployment; need observability and autoscaling.",
         intro_hint="Ask about enterprise ARR and sticky workloads.",
         hiring_index=0.45, traffic_index=0.5, founder_signal=0),

    dict(company="Railway", thesis="Developer Tools", stage="Series A",
         total_raised=20_000_000, last_round="Series A (2022)",
         notable_investors=["Placeholder"], one_liner="PaaS to deploy apps & databases with great developer UX.",
         website="https://railway.app/", sources=["https://railway.app/changelog"], why_usv="Dev productivity + bottoms-up growth.",
         why_now="Teams consolidating tooling; DX becomes a purchase driver.",
         intro_hint="Ask about paid seats growth and enterprise adoption.",
         hiring_index=0.5, traffic_index=0.6, founder_signal=0),

    dict(company="Fly.io", thesis="Developer Tools", stage="Series A",
         total_raised=12_000_000, last_round="Series A (2022)",
         notable_investors=["Placeholder"], one_liner="Deploy apps close to users on a global edge platform.",
         website="https://fly.io/", sources=["https://fly.io/blog/"], why_usv="Latency + DX = better user experience.",
         why_now="AI + realtime apps need global edges; easy deploy wins.",
         intro_hint="Ask about region usage, egress costs, and paid growth.",
         hiring_index=0.45, traffic_index=0.55, founder_signal=0),

    # --- Your original entries (kept for reference; will be filtered by defaults) ---
    dict(company="Teller", thesis="Fintech Infrastructure", stage="Seed / Early",
         total_raised=6_000_000, last_round="Seed (2022)",
         notable_investors=["Lightspeed","SciFi VC"],
         one_liner="API platform for secure access to U.S. bank data—no screen scraping.",
         website="https://www.teller.io/", sources=["https://teller.io/blog"],
         why_usv="Open finance rails; cleaner data access for consumer/SMB fintech.",
         why_now="Banks tightening; developers want direct, reliable data integrations.",
         intro_hint="Ask about bank coverage, latency/reliability SLOs, and top fintech adopters.",
         hiring_index=0.25, traffic_index=0.3, founder_signal=0),

    dict(company="Watershed", thesis="Climate Tech", stage="Series B / Growth",
         total_raised=83_000_000, last_round="Series B (2022)",
         notable_investors=["Sequoia","Kleiner Perkins"],
         one_liner="Carbon accounting & reduction platform to help enterprises hit net-zero targets.",
         website="https://www.watershed.com/", sources=["https://www.watershed.com/blog","https://techcrunch.com/"],
         why_usv="Core to climate mitigation; strong enterprise wedge with regulatory tailwinds.",
         why_now="Enterprises racing to meet reporting standards; expanding categories beyond accounting.",
         intro_hint="Ask for 3 enterprise customer references and time-to-implementation metrics.",
         hiring_index=0.6, traffic_index=0.7, founder_signal=1),

    dict(company="Span.io", thesis="Climate Tech", stage="Series C / Growth",
         total_raised=231_000_000, last_round="Series C (2023)",
         notable_investors=["Coatue","Congruent"],
         one_liner="Smart electrical panels that orchestrate home energy (solar, EV, batteries).",
         website="https://www.span.io/", sources=["https://www.span.io/blog","https://www.businesswire.com/"],
         why_usv="Hardware + software layer at the home edge; potential network effects across devices.",
         why_now="Electrification wave + DER incentives; panel as control point is gaining adoption.",
         intro_hint="Ask about install velocity, partners, and attach rates for EV/solar/batteries.",
         hiring_index=0.5, traffic_index=0.55, founder_signal=0),

    dict(company="Perplexity AI", thesis="AI / Machine Intelligence", stage="Series B / Growth",
         total_raised=165_000_000, last_round="Series B (2024)",
         notable_investors=["IVP","NEA","Jeff Bezos"],
         one_liner="AI-powered conversational search with cited answers across the live web.",
         website="https://www.perplexity.ai/", sources=["https://www.reuters.com/technology/"],
         why_usv="Aligns with 'open internet' & machine intelligence; strong engagement loops.",
         why_now="Exploding daily usage; moving from consumer curiosity to daily workflow.",
         intro_hint="Ask about retention cohorts and enterprise/education use cases.",
         hiring_index=0.9, traffic_index=0.95, founder_signal=1),

    dict(company="Hugging Face", thesis="AI / Open Source", stage="Series D / Growth",
         total_raised=235_000_000, last_round="Series D (2023)",
         notable_investors=["Sequoia","Coatue","Lux"],
         one_liner="Open-source hub for models, datasets, & tooling; the GitHub of AI.",
         website="https://huggingface.co/", sources=["https://huggingface.co/blog"],
         why_usv="Open networks & developer ecosystems; durable community moat.",
         why_now="Model hosting/inference partnerships expanding; developer pull remains strong.",
         intro_hint="Ask about paid usage mix, enterprise plans, and ecosystem monetization.",
         hiring_index=0.75, traffic_index=0.85, founder_signal=1),

    dict(company="Warp", thesis="Developer Tools", stage="Series B / Growth",
         total_raised=73_000_000, last_round="Series B (2023)",
         notable_investors=["GV","Dylan Field"],
         one_liner="Modern terminal with AI assist and multiplayer collaboration.",
         website="https://www.warp.dev/", sources=["https://www.warp.dev/blog"],
         why_usv="Developer productivity + network effects; wedge into daily workflows.",
         why_now="AI-native commands and team features are accelerating adoption.",
         intro_hint="Ask for active team usage, command share rates, and AI resolution metrics.",
         hiring_index=0.55, traffic_index=0.6, founder_signal=0),

    dict(company="Plaid Climate", thesis="Climate + Fintech", stage="Spin-out / Early",
         total_raised=None, last_round="—",
         notable_investors=["Plaid (parent)"],
         one_liner="Sustainability tools layered on transaction data for climate impact tracking.",
         website="https://plaid.com/", sources=["https://plaid.com/blog/"],
         why_usv="Intersection of open finance and climate; unique data signal for impact reporting.",
         why_now="Banks/fintechs under pressure to offer climate reporting to consumers/SMBs.",
         intro_hint="Ask about pilot partners and accuracy of emissions estimation models.",
         hiring_index=0.3, traffic_index=0.4, founder_signal=0),

    dict(company="Aleo", thesis="Open Data / Privacy Infra", stage="Series B / Growth",
         total_raised=298_000_000, last_round="Series B (2022)",
         notable_investors=["a16z","SoftBank"],
         one_liner="Zero-knowledge blockchain enabling private applications with on-chain verification.",
         website="https://www.aleo.org/", sources=["https://www.aleo.org/blog"],
         why_usv="Privacy-preserving apps for the open internet; crypto rails for agents.",
         why_now="ZK tooling maturing; developers shipping beyond POCs.",
         intro_hint="Ask about top dev projects, TVL, and onchain activity growth.",
         hiring_index=0.45, traffic_index=0.5, founder_signal=0),

    dict(company="Worldcoin", thesis="Decentralized ID", stage="Series C / Growth",
         total_raised=250_000_000, last_round="Series C (2023)",
         notable_investors=["a16z","Bain Capital Crypto"],
         one_liner="Global proof-of-personhood protocol aiming to enable equitable digital identity.",
         website="https://worldcoin.org/", sources=["https://worldcoin.org/blog"],
         why_usv="Identity for open internet & agents; unlocks fair airdrops and spam resistance.",
         why_now="Expanding verification; policy debates drive relevance and adoption questions.",
         intro_hint="Ask for verified user growth by geography and active developer integrations.",
         hiring_index=0.4, traffic_index=0.55, founder_signal=0),

    dict(company="Uniswap Labs", thesis="Open Internet / DeFi", stage="Series B / Growth",
         total_raised=176_000_000, last_round="Series B (2022)",
         notable_investors=["a16z","Paradigm"],
         one_liner="Leading decentralized exchange protocol—open, programmable liquidity.",
         website="https://uniswap.org/", sources=["https://blog.uniswap.org/"],
         why_usv="Open financial infrastructure; network effects & protocol-driven moats.",
         why_now="Onchain activity cycling up; L2 expansion and new product surfaces.",
         intro_hint="Ask about L2 share, fee capture vs protocol, and ecosystem apps growth.",
         hiring_index=0.35, traffic_index=0.6, founder_signal=1),
]

df = pd.DataFrame(data)

# ---------------------------------
# Derived fields
# ---------------------------------
df["domain"] = df["website"].apply(canonical_domain)
df["stage_norm"] = df["stage"].apply(canonical_stage)

# For Neutral mode only (we keep it for completeness)
df["recent_funding_usd"] = df.get("recent_funding_usd", pd.Series([np.nan]*len(df)))
df["recent_funding_usd"] = df["recent_funding_usd"].fillna(df["total_raised"])
df["recent_funding_usd_norm"] = normalize_series_0_1(df["recent_funding_usd"].fillna(0))

# Capital-efficiency proxy: lower raised within same stage → better
def stage_efficiency_proxy(row: pd.Series, stage_group_stats: dict) -> float:
    stg = row.get("stage_norm","unknown")
    amt = float(row.get("total_raised") or 0.0)
    lo, hi = stage_group_stats.get(stg, (0.0, 0.0))
    if hi <= lo: return 0.5
    norm = (amt - lo) / (hi - lo)
    return round(1.0 - min(max(norm, 0.0), 1.0), 3)

stage_stats = {}
for stg, g in df.groupby("stage_norm"):
    vals = g["total_raised"].dropna().astype(float)
    stage_stats[stg] = (float(vals.min()) if len(vals) else 0.0,
                        float(vals.max()) if len(vals) else 0.0)
df["capital_eff"] = df.apply(lambda r: stage_efficiency_proxy(r, stage_stats), axis=1)

# ---------------------------------
# De-dup
# ---------------------------------
dedupe_before = len(df)
df = df.sort_values(["company"]).drop_duplicates(subset=["domain"], keep="first")
df = df.drop_duplicates(subset=["company"], keep="first")
deduped_count = dedupe_before - len(df)

# ---------------------------------
# Header
# ---------------------------------
st.title("🔥 Early-Stage Hotlist (USV Core)")
st.subheader("Seed & Series A focus. Climate/Crypto excluded by default (toggle in sidebar).")
if deduped_count > 0:
    st.caption(f"De-duplicated {deduped_count} item(s) by domain/company.")

# ---------------------------------
# Session state
# ---------------------------------
if "owners" not in st.session_state: st.session_state["owners"] = {}
if "status" not in st.session_state: st.session_state["status"] = {}
if "notes_map" not in st.session_state: st.session_state["notes_map"] = {}

STATUS_OPTS = ["New", "Reviewing", "Waiting on Data", "Pass", "Advance"]

# ---------------------------------
# Sidebar filters
# ---------------------------------
with st.sidebar:
    st.header("Filter")
    thesis_opts = ["All"] + sorted(df["thesis"].unique())
    stage_opts = ["All"] + sorted(df["stage"].unique())
    pick_thesis = st.selectbox("Thesis", thesis_opts, index=0)
    pick_stage = st.selectbox("Stage", stage_opts, index=0)
    min_amt = st.slider("Min total raised ($M)", 0, int((df["total_raised"].fillna(0).max()) / 1_000_000), 0)

    st.markdown("---")
    st.subheader("Core constraints")
    early_only = st.toggle("Early-stage only (Pre-seed / Seed / Series A)", value=True)
    exclude_climate_crypto = st.toggle("Exclude Climate & Crypto/Web3 theses", value=True)

    allowed_theses = set(ALL_THESES)
    if exclude_climate_crypto:
        allowed_theses = allowed_theses - EXCLUDED_THESES_DEFAULT

    st.markdown("---")
    st.subheader("Scoring mode")
    core_mode = st.toggle("Core scoring (Seed/A bias)", value=True)

    # Optional: quick diagnostics
    show_diag = st.checkbox("Show filter diagnostics", value=False)

    # Optional ad-hoc thesis boost (applies only within allowed_theses)
    focus_theses = st.multiselect("Extra thesis boost", sorted(list(allowed_theses)), default=list(allowed_theses))

    min_score = st.slider("Min score", 0, 100, 0)
    sort_by = st.selectbox("Sort by", ["Highest score", "Largest total raised", "Company A→Z", "Most recent round label"], index=0)

    hide_pass = st.checkbox("Hide 'Pass' status", value=False)

# Filter summary banner
st.caption(
    "Filters active → "
    + ("Early-stage only; " if early_only else "")
    + ("Excluding Climate & Crypto/Web3; " if exclude_climate_crypto else "")
    + f"Allowed theses: {', '.join(sorted(list(allowed_theses)))}"
)

# ---------------------------------
# Apply filters (order matters)
# ---------------------------------
base_count = len(df)
f = df.copy()

# Optional UI thesis/stage picks
if pick_thesis != "All":
    f = f[f["thesis"] == pick_thesis]
if pick_stage != "All":
    f = f[f["stage"] == pick_stage]

# Core thesis allowlist
f = f[f["thesis"].isin(allowed_theses)]

# Early-stage only
if early_only:
    f = f[f["stage_norm"].isin(EARLY_STAGE_KEYS)]

# Amount threshold
f = f[f["total_raised"].fillna(0) >= (min_amt * 1_000_000)]

# Diagnostics
if show_diag:
    st.info(f"Started with {base_count} → After filters: {len(f)} rows")

# Empty state
if len(f) == 0:
    st.info("No companies match your Core filters. Try disabling **Early-stage only** or climate/crypto exclusion.")
    st.stop()

# ---------------------------------
# Scoring
# ---------------------------------
def growth_signal(row: pd.Series) -> float:
    h = row.get("hiring_index", np.nan)
    t = row.get("traffic_index", np.nan)
    g = float(np.nanmean([h, t]))
    return float(0.0 if np.isnan(g) else min(max(g, 0.0), 1.0))

def founder_signal01(row: pd.Series) -> float:
    fs = str(row.get("founder_signal", "")).strip().lower()
    return 1.0 if fs in ("1","true","yes","y") else 0.0

def thematic_fit01(row: pd.Series, allowed: set[str]) -> float:
    return 1.0 if row.get("thesis") in allowed else 0.0

def compute_score_core(row: pd.Series, allowed: set[str], extra_boost: list[str]) -> tuple[float, dict]:
    stage_key = row.get("stage_norm","unknown")
    stage_fit = STAGE_WEIGHTS_CORE.get(stage_key, STAGE_WEIGHTS_CORE["unknown"])
    founder = founder_signal01(row)
    thematic_base = thematic_fit01(row, allowed)
    thematic_extra = 1.0 if (extra_boost and row.get("thesis") in extra_boost) else 0.0
    thematic = max(thematic_base, thematic_extra)
    growth = growth_signal(row)
    cap_eff = float(row.get("capital_eff") or 0.5)

    w = WEIGHTS_CORE
    score01 = stage_fit*w["stage_fit"] + founder*w["founder_signal"] + thematic*w["thematic_fit"] + growth*w["growth_signal"] + cap_eff*w["capital_eff"]
    breakdown = {
        "stage_fit":       round(stage_fit * w["stage_fit"] * 100, 1),
        "founder_signal":  round(founder   * w["founder_signal"] * 100, 1),
        "thematic_fit":    round(thematic  * w["thematic_fit"] * 100, 1),
        "growth_signal":   round(growth    * w["growth_signal"] * 100, 1),
        "capital_eff":     round(cap_eff   * w["capital_eff"] * 100, 1),
    }
    return round(score01 * 100, 1), breakdown

def compute_score_neutral(row: pd.Series, allowed: set[str]) -> tuple[float, dict]:
    funding = float(row.get("recent_funding_usd_norm") or 0.0)
    growth  = growth_signal(row)
    thematic = 1.0 if row.get("thesis") in allowed else 0.0
    founder = founder_signal01(row)
    w = WEIGHTS_NEUTRAL
    score01 = funding*w["recent_funding"] + growth*w["growth_signal"] + thematic*w["thematic_fit"] + founder*w["founder_signal"]
    breakdown = {
        "recent_funding": round(funding * w["recent_funding"] * 100, 1),
        "growth_signal":  round(growth  * w["growth_signal"]  * 100, 1),
        "thematic_fit":   round(thematic* w["thematic_fit"]   * 100, 1),
        "founder_signal": round(founder * w["founder_signal"] * 100, 1),
    }
    return round(score01 * 100, 1), breakdown

if core_mode:
    f["score"], f["_breakdown"] = zip(*f.apply(lambda r: compute_score_core(r, set(allowed_theses), focus_theses), axis=1))
else:
    f["score"], f["_breakdown"] = zip(*f.apply(lambda r: compute_score_neutral(r, set(allowed_theses)), axis=1))

# Hide Pass
def row_status(name): return st.session_state["status"].get(name, "New")
if hide_pass:
    f = f[f["company"].apply(lambda n: row_status(n) != "Pass")]

# Threshold & sort
f = f[f["score"] >= min_score]
if sort_by == "Largest total raised":
    f = f.sort_values("total_raised", ascending=False, na_position="last")
elif sort_by == "Company A→Z":
    f = f.sort_values("company", ascending=True)
elif sort_by == "Most recent round label":
    f = f.sort_values("last_round", ascending=False, na_position="last")
else:
    f = f.sort_values("score", ascending=False, na_position="last")

# ---------------------------------
# Summary
# ---------------------------------
st.markdown("### Summary")
total_companies = len(f)
total_raised = f["total_raised"].sum(skipna=True)
avg_raised = f["total_raised"].mean(skipna=True) if total_companies > 0 else None
avg_score = round(float(f["score"].mean()), 1) if total_companies > 0 else None

def _flatten(xs):
    out = []
    for row in xs.dropna(): out.extend(row)
    return out

unique_investors = sorted(set(_flatten(f["notable_investors"])) - {""}) if total_companies else []
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Companies", total_companies)
c2.metric("Total raised", fmt_money(total_raised))
c3.metric("Avg total raised", fmt_money(avg_raised))
c4.metric("Notable investors", len(unique_investors))
c5.metric("Avg score", "—" if avg_score is None else f"{avg_score}")

# ---------------------------------
# Results
# ---------------------------------
st.markdown("### Companies")
if total_companies == 0:
    st.info("No companies match your filters.")
else:
    for _, r in f.iterrows():
        name = r["company"]
        with st.container(border=True):
            thesis_badge = as_badge(r["thesis"])
            stage_badge  = as_badge(r["stage"])
            money_badge  = as_badge(fmt_money(r["total_raised"]))
            score_badge  = as_badge(f"Score {r['score']}")
            st.markdown("  ".join([f"**{name}**", thesis_badge, stage_badge, money_badge, score_badge]), unsafe_allow_html=True)

            st.write(r["one_liner"])

            col1, col2, col3, col4 = st.columns([3,3,3,3])
            col1.write(f"**Last round:** {r['last_round']}")
            col2.write(f"**Website:** [{r['website']}]({r['website']})")
            invs = ", ".join(r["notable_investors"]) if isinstance(r["notable_investors"], list) else "—"
            col3.write(f"**Notable investors:** {invs if invs else '—'}")
            col4.write(f"**Thesis:** {r['thesis']}")

            with st.expander("Score breakdown"):
                bd = r["_breakdown"]
                if core_mode:
                    st.write(
                        f"- Stage fit (Seed/A bias): **{bd['stage_fit']}**"
                        f"\n- Founder signal: **{bd['founder_signal']}**"
                        f"\n- Thematic fit (allowed): **{bd['thematic_fit']}**"
                        f"\n- Growth signal: **{bd['growth_signal']}**"
                        f"\n- Capital efficiency: **{bd['capital_eff']}**"
                        f"\n\n**Total:** {r['score']}"
                    )
                else:
                    st.write(
                        f"- Recent funding: **{bd['recent_funding']}**"
                        f"\n- Growth signal: **{bd['growth_signal']}**"
                        f"\n- Thematic fit: **{bd['thematic_fit']}**"
                        f"\n- Founder signal: **{bd['founder_signal']}**"
                        f"\n\n**Total:** {r['score']}"
                    )

            st.write(f"**Why USV:** {r['why_usv']}")
            st.write(f"**Why now:** {r['why_now']}")

            if isinstance(r["sources"], list) and r["sources"]:
                links = " · ".join([f"[source]({u})" for u in r["sources"]])
                st.write(f"**Sources:** {links}")

            a1, a2, a3, a4 = st.columns([2,2,3,3])
            owner_val = st.session_state["owners"].get(name, "")
            a1.text_input("Owner", value=owner_val, key=f"owner_{name}")
            st.session_state["owners"][name] = st.session_state[f"owner_{name}"].strip()

            status_val = st.session_state["status"].get(name, "New")
            a2.selectbox("Status", ["New","Reviewing","Waiting on Data","Pass","Advance"],
                         index=["New","Reviewing","Waiting on Data","Pass","Advance"].index(status_val) if status_val in ["New","Reviewing","Waiting on Data","Pass","Advance"] else 0,
                         key=f"status_{name}")
            st.session_state["status"][name] = st.session_state[f"status_{name}"]

            note_val = st.session_state["notes_map"].get(name, "")
            a3.text_input("Short note", value=note_val, key=f"note_{name}")
            st.session_state["notes_map"][name] = st.session_state[f"note_{name}"]

            link = deep_link_to_copilot(name, r["website"])
            if link:
                a4.link_button("Open in DD Copilot", link, use_container_width=True)
            else:
                a4.caption("Set DDLITE_URL to enable Copilot deep link")

            ask = r.get("intro_hint") or "Ask for 3 customer references and latest traction metrics."
            outreach = f"Hi — exploring {name} for USV’s Core thesis. Could you intro me to the team? {ask}"
            st.write("**Next action:**")
            st.code(outreach, language="text")

# ---------------------------------
# Export
# ---------------------------------
st.subheader("Export")
export_cols = [
    "company","thesis","stage","total_raised","last_round",
    "notable_investors","one_liner","website","sources","why_usv","why_now","score"
]
exp = f[export_cols].copy()
exp["owner"]  = exp["company"].apply(lambda n: st.session_state["owners"].get(n, ""))
exp["status"] = exp["company"].apply(lambda n: st.session_state["status"].get(n, "New"))
exp["note"]   = exp["company"].apply(lambda n: st.session_state["notes_map"].get(n, ""))

csv = exp.to_csv(index=False)
st.download_button("Download CSV", csv, "usv_deal_hotlist.csv", "text/csv", use_container_width=True)

engine = EXCEL_ENGINE
if engine:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine=engine) as writer:
        exp.to_excel(writer, index=False, sheet_name="Hotlist")
    st.download_button(
        f"Download Excel ({engine})",
        buf.getvalue(),
        "usv_deal_hotlist.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
else:
    st.info("Excel export unavailable: add `XlsxWriter` or `openpyxl` to requirements.txt. CSV export works above.")
