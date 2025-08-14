# hotlist_app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from io import BytesIO
from importlib.util import find_spec

st.set_page_config(page_title="USV Deal Hotlist", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def tznow():
    return datetime.now(timezone.utc)

def days_ago(d: pd.Timestamp) -> int:
    return max(0, (pd.Timestamp(tznow().date()) - d.normalize()).days)

def fmt_money(n: int) -> str:
    if n is None: return "—"
    if n >= 1_000_000_000: return f"${n/1_000_000_000:.1f}B"
    return f"${n/1_000_000:.1f}M"

def badge(txt: str):
    return f"<span style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:999px;padding:2px 8px;font-size:12px;color:#3730a3'>{txt}</span>"

def pick_excel_engine():
    if find_spec("xlsxwriter"): return "xlsxwriter"
    if find_spec("openpyxl"): return "openpyxl"
    return None

EXCEL_ENGINE = pick_excel_engine()

# -----------------------------
# Demo dataset (curated; public facts tone)
# Fields: company, thesis, one_liner, stage, funding, hq, date, why_usv, why_now, signals[list], founders[list of {name,url}], sources[list], intro_hint
# -----------------------------
data = [
    # AI Infrastructure & Agentic Stack
    dict(
        company="GigaIO",
        thesis="AI Infrastructure & Agentic Stack",
        one_liner="Composable AI inferencing hardware; suitcase‑size edge ‘supernodes’.",
        stage="Series B",
        funding=21_000_000,
        hq="San Diego, CA",
        date="2025-07-18",
        why_usv="Infrastructure that reduces cost per inference and enables distributed, energy‑aware agent workloads.",
        why_now="Fresh round; shipping edge units. Customers piloting on constrained power footprints.",
        signals=["New funding", "Product shipping", "Infra cost leverage"],
        founders=[dict(name="Alan Benjamin", url="")],
        sources=["https://gigaio.com/"],
        intro_hint="Start with hardware partnerships/customer references; ask for data center pilot outcomes."
    ),
    dict(
        company="fal",
        thesis="AI Infrastructure & Agentic Stack",
        one_liner="Hosted inference + infra for production AI workloads.",
        stage="Series C",
        funding=125_000_000,
        hq="Remote / US",
        date="2025-07-31",
        why_usv="Agentic apps need reliable, cost‑efficient inference at scale; aligns with USV’s machine intelligence stack view.",
        why_now="Large round; likely expanding enterprise GTM and ecosystem integrations.",
        signals=["New funding", "Enterprise traction"],
        founders=[dict(name="Team fal", url="https://fal.ai/")],
        sources=["https://fal.ai/"],
        intro_hint="Probe enterprise design partners; ask about price/perf vs hyperscalers."
    ),
    dict(
        company="TensorWave",
        thesis="AI Infrastructure & Agentic Stack",
        one_liner="AMD‑accelerated GPU clusters for model training.",
        stage="Series A",
        funding=100_000_000,
        hq="Austin, TX",
        date="2025-05-14",
        why_usv="Alt‑GPU supply and cost curves matter for open models; potential wedge for vertical model providers.",
        why_now="Large A; supply access amidst GPU scarcity makes them a near‑term enabler.",
        signals=["New funding", "Capacity expansion"],
        founders=[dict(name="Mihir Vaidya", url="")],
        sources=["https://www.tensorwave.com/"],
        intro_hint="Ask about committed capacity and utilization; references among model builders."
    ),

    # Vertical / Niche AI
    dict(
        company="Hume AI",
        thesis="Vertical/Niche AI",
        one_liner="Affective models for natural human‑AI interaction.",
        stage="Series B",
        funding=50_000_000,
        hq="New York, NY",
        date="2025-01-22",
        why_usv="USV views emotion/sentiment as a durable edge for agents in consumer and health contexts.",
        why_now="SDK adoption rising; ecosystem exploring ‘empathetic agents’.",
        signals=["SDK adoption", "Developer momentum"],
        founders=[dict(name="Alan Cowen", url="https://www.hume.ai/")],
        sources=["https://www.hume.ai/"],
        intro_hint="Ask for top production use cases; latency/accuracy on speech vs. text."
    ),
    dict(
        company="Endex",
        thesis="Vertical/Niche AI",
        one_liner="Agentic workflows inside spreadsheets for analysts.",
        stage="Series A",
        funding=14_000_000,
        hq="San Francisco, CA",
        date="2025-08-01",
        why_usv="‘Software that works for you’ in a ubiquitous surface area (Excel) aligns with USV’s practical agents thesis.",
        why_now="Fresh round; early GTM to finance teams is moving now.",
        signals=["New funding", "B2B agent traction"],
        founders=[dict(name="Team Endex", url="https://endex.ai/")],
        sources=["https://endex.ai/"],
        intro_hint="Ask for 3 reference accounts and measure manual work reduction %."
    ),

    # Climate & Real‑World Systems
    dict(
        company="Amogy",
        thesis="Climate & Real‑World Systems",
        one_liner="Ammonia‑to‑power for ships and data centers.",
        stage="Growth",
        funding=23_000_000,
        hq="Brooklyn, NY",
        date="2025-07-15",
        why_usv="Bridges bits ↔ atoms for AI energy demand; decarbonization pathway with infra tie‑ins.",
        why_now="Data center energy crunch; maritime pilots underway.",
        signals=["Pilots", "DC energy relevance"],
        founders=[dict(name="Seonghoon Woo", url="https://amogy.co/")],
        sources=["https://amogy.co/"],
        intro_hint="Ask for DC pilot metrics and cost parity timelines."
    ),
    dict(
        company="Pano AI",
        thesis="Climate & Real‑World Systems",
        one_liner="AI wildfire detection at scale.",
        stage="Series B",
        funding=44_000_000,
        hq="San Francisco, CA",
        date="2025-07-10",
        why_usv="Adaptation tech with recurring municipal/utility revenue; strong data moats.",
        why_now="Wildfire seasons intensifying; expanding covered acreage.",
        signals=["New funding", "Public sector wins"],
        founders=[dict(name="Sonia Kastner", url="https://www.pano.ai/")],
        sources=["https://www.pano.ai/"],
        intro_hint="Request utility references and incident detection precision/recall."
    ),

    # Open Data / Ownership / Crypto-as-Rails
    dict(
        company="Privy",
        thesis="Open Data / Ownership / Crypto-as-Rails",
        one_liner="Identity + wallet infra for consumer apps and agents.",
        stage="Series A",
        funding=18_000_000,
        hq="Remote / US",
        date="2025-06-30",
        why_usv="User‑owned data + agent payments needs wallets/identity without UX tax.",
        why_now="Agent apps seeking embedded wallets; developer demand rising.",
        signals=["Developer momentum", "Ecosystem integrations"],
        founders=[dict(name="Henri", url="https://www.privy.io/")],
        sources=["https://www.privy.io/"],
        intro_hint="Ask for time‑to‑integrate and retention of active wallets."
    ),
    dict(
        company="Farcaster",
        thesis="Open Data / Ownership / Crypto-as-Rails",
        one_liner="Open social protocol and data portability primitives.",
        stage="Series A",
        funding=30_000_000,
        hq="San Francisco, CA",
        date="2025-05-20",
        why_usv="Open data networks are core to USV; Farcaster powers identity/graph for agents.",
        why_now="Ecosystem growth; developer apps and channels scaling.",
        signals=["Open protocol adoption", "Developer ecosystem"],
        founders=[dict(name="Dan Romero", url="https://www.farcaster.xyz/")],
        sources=["https://www.farcaster.xyz/"],
        intro_hint="Ask for top 10 third‑party apps and their growth metrics."
    ),

    # Adjacent dev tools
    dict(
        company="Zed",
        thesis="Developer Tools",
        one_liner="High‑performance collaborative code editor.",
        stage="Series A",
        funding=12_000_000,
        hq="San Francisco, CA",
        date="2024-07-20",
        why_usv="Collaborative dev surfaces enable network effects; complements AI coding agents.",
        why_now="Growing team/collab features; plugin ecosystem forming.",
        signals=["OSS/community", "Collab features"],
        founders=[dict(name="Zed Team", url="https://zed.dev/")],
        sources=["https://zed.dev/"],
        intro_hint="Ask about active teams, plugin usage, and daily collaborative sessions."
    ),
]

df = pd.DataFrame(data)
df["Date"] = pd.to_datetime(df["date"])
df["DaysAgo"] = df["Date"].apply(days_ago)

# Priority score: recent (weight 2) + #signals (1) + funding scale (0.5)
df["SignalCount"] = df["signals"].apply(lambda xs: len(xs) if isinstance(xs, list) else 0)
df["Priority"] = (df["SignalCount"] * 1.0) + (df["funding"] / 50_000_000 * 0.5) + ( (90 - df["DaysAgo"]).clip(lower=0) / 90 * 2.0 )

# ---------------------------------
# Header / Value prop
# ---------------------------------
st.title("🔥 USV Deal Hotlist")
st.subheader("Thesis‑aligned companies to move on now")
st.caption("Why USV? Why now? What’s the next action. Curated public signals; no proprietary data.")

# Summary bar with narrative
total_companies = len(df)
last_30 = (df["DaysAgo"] <= 30).sum()
total_funding = df["funding"].sum()
avg_round = df["funding"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Companies on hotlist", total_companies)
c2.metric("Raised in last 30 days", int(last_30))
c3.metric("Total funding", fmt_money(int(total_funding)))
c4.metric("Avg recent round", fmt_money(int(avg_round)) if avg_round == avg_round else "—")

st.markdown(
    "These are **USV‑fit** opportunities ranked by **recency**, **signal strength** (funding, launches, hiring), and **scale**. "
    "Each card tells you why it matters for USV and the next step to take."
)

# ---------------------------------
# Filters
# ---------------------------------
with st.sidebar:
    st.header("Filter")
    thesis_opts = ["All"] + sorted(df["thesis"].unique())
    stage_opts = ["All"] + sorted(df["stage"].unique())
    pick_thesis = st.selectbox("Thesis", thesis_opts, index=0)
    pick_stage = st.selectbox("Stage", stage_opts, index=0)
    min_amt = st.slider("Min funding ($M)", 0, int(df["funding"].max()/1_000_000), 0)
    recency = st.selectbox("Time window", ["All", "Last 30 days", "Last 90 days"], index=1)
    sort_by = st.selectbox("Sort by", ["Priority", "Most recent", "Largest round"], index=0)

def within_window(days, window):
    if window == "All": return True
    if window == "Last 30 days": return days <= 30
    if window == "Last 90 days": return days <= 90
    return True

f = df.copy()
if pick_thesis != "All":
    f = f[f["thesis"] == pick_thesis]
if pick_stage != "All":
    f = f[f["stage"] == pick_stage]
f = f[f["funding"] >= (min_amt * 1_000_000)]
f = f[f["DaysAgo"].apply(lambda d: within_window(d, recency))]

if sort_by == "Priority":
    f = f.sort_values(["Priority","Date"], ascending=[False, False])
elif sort_by == "Most recent":
    f = f.sort_values("Date", ascending=False)
else:
    f = f.sort_values("funding", ascending=False)

# ---------------------------------
# Results
# ---------------------------------
st.markdown("### Opportunities")
if len(f) == 0:
    st.info("No companies match your filters.")
else:
    for _, r in f.iterrows():
        with st.container(border=True):
            # Title row
            title = f"**{r['company']}**  {badge(r['thesis'])}  {badge(r['stage'])}  {badge(f\"{fmt_money(int(r['funding']))}\")}"
            st.markdown(title, unsafe_allow_html=True)
            st.write(r["one_liner"])

            # Detail row
            d1, d2, d3, d4 = st.columns([3,3,2,2])
            d1.write(f"**Why USV:** {r['why_usv']}")
            d2.write(f"**Why now:** {r['why_now']}")
            d3.write(f"**HQ:** {r['hq']}")
            d4.write(f"**When:** {r['Date'].date().isoformat()}  ({r['DaysAgo']} days ago)")

            # Signals
            if isinstance(r["signals"], list) and r["signals"]:
                st.write("**Signals:** " + " · ".join([f"• {s}" for s in r["signals"]]))

            # Founders & sources
            cols = st.columns([3,5])
            with cols[0]:
                if isinstance(r["founders"], list) and r["founders"]:
                    st.write("**Founders:**")
                    for fnd in r["founders"]:
                        if fnd.get("url"):
                            st.markdown(f"- [{fnd['name']}]({fnd['url']})")
                        else:
                            st.write(f"- {fnd['name']}")
            with cols[1]:
                if isinstance(r["sources"], list) and r["sources"]:
                    st.write("**Sources:** " + " · ".join([f"[link]({u})" for u in r["sources"]]))

            # Next action with copyable outreach
            st.write("**Next action:**")
            ask = r["intro_hint"] or "Ask for 3 design partner references and recent traction metrics."
            outreach = f"Hi — exploring {r['company']} for our thesis. Could you intro me to the team? {ask}"
            st.code(outreach, language="text")

# ---------------------------------
# Export
# ---------------------------------
st.subheader("Export")
export_cols = ["company","thesis","one_liner","stage","funding","hq","date","why_usv","why_now","signals","sources"]
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
