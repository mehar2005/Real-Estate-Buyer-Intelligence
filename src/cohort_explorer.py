import pandas as pd
import plotly.express as px
import streamlit as st


def _options(df, col):
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().astype(str).unique().tolist())


def render_cohort_explorer(result):
    st.header("Interactive Buyer Cohort Explorer")
    st.write("Create a reusable cohort across cluster assignment and available buyer attributes, then inspect the filtered profile and export the matching records.")
    df = result["clustered"].copy()
    filters = [c for c in ["cluster", "country", "region", "client_type", "acquisition_purpose", "loan_applied", "referral_channel"] if c in df.columns]
    with st.form("cohort_filters"):
        cols = st.columns(2)
        selections = {}
        for i, col in enumerate(filters):
            with cols[i % 2]:
                opts = _options(df, col)
                selections[col] = st.multiselect(col.replace("_", " ").title(), opts, default=[], key=f"cohort_{col}")
        min_satisfaction = None
        if "satisfaction_score" in df.columns and pd.to_numeric(df["satisfaction_score"], errors="coerce").notna().any():
            vals = pd.to_numeric(df["satisfaction_score"], errors="coerce")
            low, high = float(vals.min()), float(vals.max())
            min_satisfaction = st.slider("Minimum satisfaction score", low, high, low, key="cohort_satisfaction")
        submitted = st.form_submit_button("Apply cohort filters")

    # Apply defaults on first render as well; form submission is useful for explicit filtering intent.
    filtered = df.copy()
    for col, chosen in selections.items():
        if chosen:
            filtered = filtered[filtered[col].astype(str).isin(chosen)]
    if min_satisfaction is not None:
        filtered = filtered[pd.to_numeric(filtered["satisfaction_score"], errors="coerce") >= min_satisfaction]

    a, b, c = st.columns(3)
    a.metric("Matching buyers", f"{len(filtered):,}")
    b.metric("Share of all buyers", f"{len(filtered) / len(df) * 100:.1f}%" if len(df) else "0%")
    active_filters = sum(bool(v) for v in selections.values())
    if min_satisfaction is not None and "satisfaction_score" in df.columns:
        all_scores = pd.to_numeric(df["satisfaction_score"], errors="coerce").dropna()
        if not all_scores.empty and min_satisfaction > float(all_scores.min()):
            active_filters += 1
    c.metric("Active filters", active_filters)

    if filtered.empty:
        st.info("No buyers match these filters. Widen or clear one or more filters.")
        return
    if "cluster" in filtered.columns:
        counts = filtered["cluster"].astype(str).value_counts().sort_index().rename_axis("Cluster").reset_index(name="Buyers")
        st.plotly_chart(px.bar(counts, x="Cluster", y="Buyers", text="Buyers", title="Cohort composition by cluster"), use_container_width=True)
    numeric = filtered.select_dtypes(include="number")
    if not numeric.empty:
        st.subheader("Numeric cohort summary")
        st.dataframe(numeric.describe().T.round(2), use_container_width=True)
    st.subheader("Matching buyer records")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Export filtered cohort", filtered.to_csv(index=False).encode("utf-8"),
                       file_name="buyer_cohort_export.csv", mime="text/csv")
    st.caption("This is a descriptive filtering tool. It does not infer causality or predict a buyer's future behavior.")
