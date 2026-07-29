# Project 1: Marketing Campaign Performance Analytics Dashboard

A PMG-aligned analytics project that demonstrates:
- KPI analysis (**CTR, CPC, CPA, ROAS, Conversion Rate**)
- Customer segmentation (**RFM + K-Means**)
- A/B test statistical validation (**Two-proportion z-test**)
- Business recommendation workflow in a Streamlit app

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Mac/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## Dataset
- You can upload your own CSV with required columns:
`channel, segment, impressions, clicks, conversions, spend, revenue, customer_id, recency_days, frequency, monetary, variant`
- Or click **Use sample dataset** inside the app.
