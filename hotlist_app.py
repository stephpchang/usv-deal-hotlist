# hotlist_app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from io import BytesIO
from importlib.util import find_spec

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="USV Deal Hotlist", layout="wide")

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

# -----------------------------
# Curated demo dataset (public-info tone)
# Fields: company, thesis, stage, total_raised, last_round, notable_investors[list],
#         one_liner, website, sources[list], why_usv, why_now, intro_hint
# -----------------------------
data = [
    dict(
        company="Watershed",
        thesis="Climate Tech",
        stage="Series B / Growth",
        total_raised=83_000_000,
        last_round="Series B (2022)",
        notable_investors=["Sequoia", "Kleiner Perkins"],
        one_liner="Carbon accounting & reduction platform to help enterprises hit net-zero targets.",
        website="https://www.watershed.com/",
        sources=["https://www.watershed.com/blog", "https://techcrunch.com/"],
        why_usv="Core to climate mitigation; strong enterprise wedge with regulatory tailwinds.",
        why_now="Enterprises racing to meet reporting standards; expanding categories beyond accounting.",
        intro_hint="Ask for 3 enterprise customer references and time-to-implementation metrics."
    ),
    dict(
        company="Span.io",
        thesis="Climate Tech",
        stage="Series C / Growth",
        total_raised=231_000_000,
        last_round="Series C (2023)",
        notable_investors=["Coatue", "Congruent"],
        one_liner="Smart electrical panels that orchestrate home energy (solar, EV, batteries).",
        website="https://www.span.io/",
        sources=["https://www.span.io/blog", "https://www.businesswire.com/"],
        why_usv="Hardware + software layer at the home edge; potential network effects across devices.",
        why_now="Electrification wave + DER incentives; panel as control point is gaining adoption.",
        intro_hint="Ask about install velocity, partners, and attach rates for EV/solar/batteries."
    ),
    dict(
        company="Perplexity AI",
        thesis="AI / Machine Intelligence",
        stage="Series B / Growth",
        total_raised=165_000_000,
        last_round="Series B (2024)",
        notable_investors=["IVP", "NEA", "Jeff Bezos"],
        one_liner="AI-powered conversational search with cited answers across the live web.",
        website="https://www.perplexity.ai/",
        sources=["https://www.reuters.com/technology/"],  # placeholder source
        why_usv="Aligns with 'open internet' & machine intelligence; strong engagement loops.",
        why_now="Exploding daily usage; moving from consumer curiosity to daily workflow.",
        intro_hint="Ask about retention cohorts and enterprise/education use cases."
    ),
    dict(
        company="Hugging Face",
        thesis="AI / Open Source",
        stage="Series D / Growth",
        total_raised=235_000_000,
        last_round="Series D (2023)",
        notable_investors=["Sequoia", "Coatue", "Lux"],
        one_liner="Open-source hub for models, datasets, & tooling; the GitHub of AI.",
        website="https://huggingface.co/",
        sources=["https://huggingface.co/blog"],
        why_usv="Open networks & developer ecosystems; durable community moat.",
        why_now="Model hosting/inference partnerships expanding; developer pull remains strong.",
        intro_hint="Ask about paid usage mix, enterprise plans, and ecosystem monetization."
    ),
    dict(
        company="Warp",
        thesis="Developer Tools",
        stage="Series B / Growth",
        total_raised=73_000_000,
        last_round="Series B (2023)",
        notable_investors=["GV", "Dylan Field"],
        one_liner="Modern terminal with AI assist and multiplayer collaboration.",
        website="https://www.warp.dev/",
        sources=["https://www.warp.dev/blog"],
        why_usv="Developer productivity + network effects; wedge into daily workflows.",
        why_now="AI-native commands and team features are accelerating adoption.",
        intro_hint="Ask for active team usage, command share rates, and AI resolution metrics."
    ),
    dict(
        company="Teller",
        thesis="Fintech Infrastructure",
        stage="Seed / Early",
        total_raised=6_000_000,
        last_round="Seed (2022)",
        notable_investors=["Lightspeed", "SciFi VC"],
        one_liner="API platform for secure access to U.S. bank data—no screen scraping.",
        website="https://www.teller.io/",
        sources=["https://teller.io/blog"],
        why_usv="Open finance rails; cleaner data access for consumer/SMB fintech.",
        why_now="Banks tightening; developers want direct, reliable data integrations.",
        intro_hint="Ask about bank coverage, latency/reliability SLOs, and top fintech adopters."
    ),
    dict(
        company="Plaid Climate",
        thesis="Climate + Fintech",
        stage="Spin-out / Early",
        total_raised=None,
        last_round="—",
        notable_investors=["Plaid (parent)"],
        one_liner="Sustainability tools layered on transaction data for climate impact tracking.",
        website="https://plaid.com/",
        sources=["https://plaid.com/blog/"],
        why_usv="Intersection of open finance and climate; unique data signal for impact reporting.",
        why_now="Banks/fintechs under pressure to offer climate reporting to consumers/SMBs.",
        intro_hint="Ask about pilot partners and accuracy of emissions estimation models."
    ),
    dict(
        company="Aleo",
        thesis="Open Data / Privacy Infra",
        stage="Series B / Growth",
        total_raised=298_000_000,
        last_round="Series B (2022)",
        notable_investors=["a16z", "SoftBank"],
        one_liner="Zero-knowledge blockchain enabling private applications with on-chain verification.",
        website="https://www.aleo.org/",
        sources=["https://www.aleo.org/blog"],
        why_usv="Privacy-preserving apps for the open internet; crypto rails for agents.",
        why_now="ZK tooling maturing; developers shipping beyond POCs.",
        intro_hint="Ask about top dev projects, TVL, and onchain activity growth."
    ),
    dict(
        company="Worldcoin",
        thesis="Decentralized ID",
        stage="Series C / Growth",
        total_raised=250_000_000,
        last_round="Series C (2023)",
        notable_investors=["a16z", "Bain Capital Crypto"],
        one_liner="Global proof-of-personhood protocol aiming to enable equitable digital identity.",
        website="https://worldcoin.org/",
        sources=["https://worldcoin.org/blog"],
        why_usv="Identity for open internet & agents; unlocks fair airdrops and spam resistance.",
        why_now="Expanding verification; policy debates drive relevance and adoption questions.",
        intro_hint="Ask for verified user growth by geography and active developer integrations."
    ),
    dict(
        company="Uniswap Labs",
        thesis="Open Internet / DeFi",
        stage="Series B / Growth",
        total_raised=176_000_000,
        last_round="Series B (2022)",
        notable_investors=["a16z", "Paradigm"],
        one_liner="Leading decentralized exchange protocol—open, programmable liquidity.",
        website="https://uniswap.org/",
        sources=["https://blog.uniswap.org/"],
        why_usv="Open financial infrastructure; network effects & protocol-driven moats.",
        why_now="Onchain activity cycling up; L2 expansion and new product surfaces.",
        intro_hint="Ask about L2 share, fee capture vs protocol, and ecosystem apps growth."
    ),
]

df = pd.DataFrame(data)

# -----------------------------
# Header / Value prop
# -----------------------------
st.title("🔥 USV Deal Hotlist")
st.subheader("Curated companies aligned with USV’s theses — why they matter now and what to do next.")
st.caption("Demo uses public information and curated notes. No proprietary data or paid APIs.")

# -----------------------------
# Filters
# -----------------------------
with st.sidebar:
    st.header("Filter")
    thesis_opts = ["All"] + sorted(df["thesis"].unique())
    stage_opts = ["All"] + sorted(df["stage"].unique())
    pick_thesis = st.selectbox("Thesis", thesis_opts, index=0)
    pick_stage = st.selectbox("Stage", stage_opts, index=0)
    min_amt = st.slider("Min total raised ($M)", 0, int((df["total_raised"].fillna(0).max()) / 1_000_000), 0)
    sort_by = st.selectbox("Sort by", ["Most recent round label", "Largest total raised", "Company A→Z"], index=0)

f = df.copy()
if pick_thesis != "All":
    f = f[f["thesis"] == pick_thesis]
if pick_stage != "All":
    f = f[f["stage"] == pick_stage]
f = f[f["total_raised"].fillna(0) >= (min_amt * 1_000_000)]

# Sorting (best-effort: we don't have dates, so we sort by label/amount/name)
if sort_by == "Largest total raised":
    f = f.sort_values("total_raised", ascending=False, na_position="last")
elif sort_by == "Company A→Z":
    f = f.sort_values("company", ascending=True)
else:
    # "Most recent round label" – keep as entered (or sort by stage text)
    f = f.sort_values("last_round", ascending=False)

# -----------------------------
# Summary metrics (quick story)
# -----------------------------
total_companies = len(f)
total_raised = f["total_raised"].sum(skipna=True)
avg_raised = f["total_raised"].mean(skipna=True) if total_companies > 0 else None

# Unique notable investors across the filtered set
def _flatten(xs):
    out = []
    for row in xs.dropna():
        out.extend(row)
    return out

unique_investors = sorted(set(_flatten(f["notable_investors"])) - {""}) if total_companies else []
st.markdown("### Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Companies", total_companies)
c2.metric("Total raised", fmt_money(total_raised))
c3.metric("Avg total raised", fmt_money(avg_raised))
c4.metric("Notable investors", len(unique_investors))

# -----------------------------
# Results
# -----------------------------
st.markdown("### Companies")
if total_companies == 0:
    st.info("No companies match your filters.")
else:
    for _, r in f.iterrows():
        with st.container(border=True):
            # Title line with badges (avoid nested f-strings)
            thesis_badge = as_badge(r["thesis"])
            stage_badge = as_badge(r["stage"])
            money_badge = as_badge(fmt_money(r["total_raised"]))
            title_line = "  ".join([f"**{r['company']}**", thesis_badge, stage_badge, money_badge])
            st.markdown(title_line, unsafe_allow_html=True)

            # One-liner
            st.write(r["one_liner"])

            # Key facts
            col1, col2, col3, col4 = st.columns([3, 3, 3, 3])
            col1.write(f"**Last round:** {r['last_round']}")
            col2.write(f"**Website:** [{r['website']}]({r['website']})")
            invs = ", ".join(r["notable_investors"]) if isinstance(r["notable_investors"], list) else "—"
            col3.write(f"**Notable investors:** {invs if invs else '—'}")
            col4.write(f"**Thesis:** {r['thesis']}")

            # Why USV / Why now
            st.write(f"**Why USV:** {r['why_usv']}")
            st.write(f"**Why now:** {r['why_now']}")

            # Sources
            if isinstance(r["sources"], list) and r["sources"]:
                links = " · ".join([f"[source]({u})" for u in r["sources"]])
                st.write(f"**Sources:** {links}")

            # Next action: copyable outreach note
            ask = r.get("intro_hint") or "Ask for 3 customer references and latest traction metrics."
            outreach = f"Hi — exploring {r['company']} for USV’s thesis. Could you intro me to the team? {ask}"
            st.write("**Next action:**")
            st.code(outreach, language="text")

# -----------------------------
# Export
# -----------------------------
st.subheader("Export")
export_cols = [
    "company", "thesis", "stage", "total_raised", "last_round",
    "notable_investors", "one_liner", "website", "sources", "why_usv", "why_now"
]
csv = f[export_cols].to_csv(index=False)
st.download_button("Download CSV", csv, "usv_deal_hotlist.csv", "text/csv", use_container_width=True)

if EXCEL_ENGINE:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine=EXCEL_ENGINE) as writer:
        f[export_cols].to_excel(writer, index=False, sheet_name="Hotlist")
    st.download_button(
        f"Download Excel ({EXCEL_ENGINE})",
        buf.getvalue(),
        "usv_deal_hotlist.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
else:
    st.info("Excel export unavailable: add `XlsxWriter` or `openpyxl` to requirements.txt. CSV export works above.")
