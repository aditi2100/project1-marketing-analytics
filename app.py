import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Data generation (sample)
# -----------------------------
def generate_sample_data(n=12000, seed=42):
    np.random.seed(seed)

    channels = np.random.choice(
        ["Google Ads", "Meta Ads", "LinkedIn", "TikTok"],
        size=n, p=[0.35, 0.30, 0.20, 0.15]
    )
    segments = np.random.choice(
        ["SMB", "Mid-Market", "Enterprise"],
        size=n, p=[0.5, 0.35, 0.15]
    )
    variants = np.random.choice(["A", "B"], size=n, p=[0.5, 0.5])

    impressions = np.random.poisson(lam=1500, size=n) + 100

    base_ctr = np.where(channels == "Google Ads", 0.040,
               np.where(channels == "Meta Ads", 0.032,
               np.where(channels == "LinkedIn", 0.026, 0.030)))
    ctr = np.clip(base_ctr + np.where(variants == "B", 0.003, 0.0) + np.random.normal(0, 0.004, n), 0.005, 0.20)
    clicks = np.maximum((impressions * ctr).astype(int), 1)

    base_cvr = np.where(segments == "Enterprise", 0.14,
               np.where(segments == "Mid-Market", 0.10, 0.08))
    cvr = np.clip(base_cvr + np.random.normal(0, 0.01, n), 0.01, 0.4)
    conversions = np.minimum((clicks * cvr).astype(int), clicks)

    cpc = np.where(channels == "LinkedIn", np.random.uniform(4, 9, n),
          np.where(channels == "Google Ads", np.random.uniform(2, 6, n),
          np.where(channels == "Meta Ads", np.random.uniform(1, 4, n), np.random.uniform(1, 5, n))))
    spend = clicks * cpc

    aov = np.where(segments == "Enterprise", np.random.uniform(250, 600, n),
          np.where(segments == "Mid-Market", np.random.uniform(120, 300, n), np.random.uniform(60, 180, n)))
    revenue = conversions * aov

    customer_id = np.random.randint(1000, 4000, size=n)
    recency_days = np.random.randint(1, 181, size=n)
    frequency = np.random.randint(1, 25, size=n)
    monetary = np.round(np.random.gamma(shape=2.5, scale=120, size=n), 2)

    return pd.DataFrame({
        "channel": channels,
        "segment": segments,
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "spend": np.round(spend, 2),
        "revenue": np.round(revenue, 2),
        "customer_id": customer_id,
        "recency_days": recency_days,
        "frequency": frequency,
        "monetary": monetary,
        "variant": variants
    })


# -----------------------------
# KPI + stats helpers
# -----------------------------
def safe_div(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    out = np.zeros_like(a, dtype=float)
    np.divide(a, b, out=out, where=b != 0)
    return out


def add_kpis(df):
    out = df.copy()
    out["ctr"] = safe_div(out["clicks"], out["impressions"])
    out["cpc"] = safe_div(out["spend"], out["clicks"])
    out["cpa"] = safe_div(out["spend"], out["conversions"])
    out["roas"] = safe_div(out["revenue"], out["spend"])
    out["conversion_rate"] = safe_div(out["conversions"], out["clicks"])
    return out


def channel_summary(df):
    g = df.groupby("channel", as_index=False).agg(
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        conversions=("conversions", "sum"),
        spend=("spend", "sum"),
        revenue=("revenue", "sum"),
    )
    return add_kpis(g).sort_values("roas", ascending=False)


def segment_rfm(df, n_clusters=4):
    rfm = df.groupby("customer_id", as_index=False).agg(
        recency_days=("recency_days", "min"),
        frequency=("frequency", "sum"),
        monetary=("monetary", "sum"),
    )
    x = rfm[["recency_days", "frequency", "monetary"]].copy()
    x_scaled = StandardScaler().fit_transform(x)
    rfm["segment"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(x_scaled)
    return rfm


def two_prop_ztest(a_success, a_total, b_success, b_total):
    p1 = a_success / a_total
    p2 = b_success / b_total
    p_pool = (a_success + b_success) / (a_total + b_total)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / a_total + 1 / b_total))
    z = (p2 - p1) / se_pool
    p_val = 2 * (1 - norm.cdf(abs(z)))

    se_unpooled = math.sqrt((p1 * (1 - p1) / a_total) + (p2 * (1 - p2) / b_total))
    margin = 1.96 * se_unpooled
    diff = p2 - p1

    return {
        "a_rate": p1,
        "b_rate": p2,
        "absolute_lift": diff,
        "relative_lift_pct": ((diff / p1) * 100) if p1 > 0 else 0.0,
        "p_value": p_val,
        "ci_95": [diff - margin, diff + margin],
        "significant_at_0_05": p_val < 0.05
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Marketing Campaign Analytics", layout="wide")
st.title("📊 Marketing Campaign Performance Analytics Dashboard")

st.write("Upload a CSV or use generated sample data.")

uploaded = st.file_uploader("Upload campaign CSV", type=["csv"])
use_sample = st.button("Use sample dataset")

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = generate_sample_data()
else:
    st.info("Click **Use sample dataset** to see output instantly.")
    st.stop()

required = [
    "channel", "segment", "impressions", "clicks", "conversions",
    "spend", "revenue", "customer_id", "recency_days", "frequency",
    "monetary", "variant"
]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

df = add_kpis(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Spend", f"${df['spend'].sum():,.0f}")
c2.metric("Total Revenue", f"${df['revenue'].sum():,.0f}")
c3.metric("Overall ROAS", f"{(df['revenue'].sum()/df['spend'].sum()):.2f}")
c4.metric("Total Conversions", f"{int(df['conversions'].sum()):,}")

st.subheader("Channel Performance")
ch = channel_summary(df)
fig = px.bar(ch, x="channel", y="roas", text=ch["roas"].round(2), color="channel")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(ch, use_container_width=True)

st.subheader("Customer Segmentation (RFM + K-Means)")
seg = segment_rfm(df, n_clusters=4)
seg_summary = seg.groupby("segment", as_index=False).agg(
    customers=("customer_id", "count"),
    avg_recency=("recency_days", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean"),
).sort_values("avg_monetary", ascending=False)
st.dataframe(seg_summary, use_container_width=True)

st.subheader("A/B Test (Conversion Rate)")
a = df[df["variant"] == "A"]
b = df[df["variant"] == "B"]
res = two_prop_ztest(
    int(a["conversions"].sum()), int(a["clicks"].sum()),
    int(b["conversions"].sum()), int(b["clicks"].sum())
)
st.json({
    "A_conversion_rate": round(res["a_rate"], 4),
    "B_conversion_rate": round(res["b_rate"], 4),
    "absolute_lift_B_minus_A": round(res["absolute_lift"], 4),
    "relative_lift_pct": round(res["relative_lift_pct"], 2),
    "p_value": round(res["p_value"], 6),
    "ci_95_diff": [round(res["ci_95"][0], 4), round(res["ci_95"][1], 4)],
    "significant_at_0_05": res["significant_at_0_05"]
})

if res["significant_at_0_05"] and res["absolute_lift"] > 0:
    st.success("Recommendation: Reallocate incremental budget toward Variant B / better-performing channels.")
elif res["significant_at_0_05"] and res["absolute_lift"] < 0:
    st.warning("Recommendation: Keep budget with Variant A; B underperforms significantly.")
else:
    st.info("Recommendation: No significant winner yet; continue test for more data.")
