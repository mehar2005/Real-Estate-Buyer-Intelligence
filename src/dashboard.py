import pandas as pd
import streamlit as st
import plotly.express as px

def render_analytics_dashboard(result):
    st.header("Interactive Analytics Dashboard")
    st.caption("Filter the client segments to explore differences in buyer behavior and linked property activity.")

    data = result["clustered"].copy()
    with st.expander("Dashboard filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        filter_cols = [
            ("country", c1, "Country"),
            ("region", c2, "Region"),
            ("acquisition_purpose", c3, "Acquisition purpose"),
            ("client_type", c4, "Client type"),
        ]
        filters = {}
        for col, container, label in filter_cols:
            with container:
                if col in data.columns:
                    vals = sorted(data[col].dropna().astype(str).unique().tolist())
                    filters[col] = st.multiselect(label, vals, default=[])
                else:
                    filters[col] = []

    filtered = data.copy()
    for col, values in filters.items():
        if values and col in filtered.columns:
            filtered = filtered[filtered[col].astype(str).isin(values)]

    if filtered.empty:
        st.warning("No clients match the selected filters. Clear one or more filters.")
        return

    a, b, c, d = st.columns(4)
    a.metric("Filtered clients", f"{len(filtered):,}")
    b.metric("Segments present", int(filtered["cluster"].nunique()))
    b2 = filtered["acquisition_purpose"].nunique(dropna=True) if "acquisition_purpose" in filtered else None
    c.metric("Acquisition purposes", b2 if b2 is not None else "—")
    d.metric("Avg. satisfaction", f"{filtered['satisfaction_score'].mean():.2f}" if "satisfaction_score" in filtered and filtered["satisfaction_score"].notna().any() else "N/A")

    left, right = st.columns(2)
    with left:
        counts = filtered["cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Clients")
        st.plotly_chart(px.bar(counts, x="Cluster", y="Clients", text="Clients", title="Filtered cluster distribution"), use_container_width=True)
    with right:
        if "acquisition_purpose" in filtered:
            cross = pd.crosstab(filtered["cluster"], filtered["acquisition_purpose"].fillna("Missing")).reset_index()
            long = cross.melt(id_vars="cluster", var_name="Acquisition purpose", value_name="Clients")
            st.plotly_chart(px.bar(long, x="cluster", y="Clients", color="Acquisition purpose",
                                   barmode="group", title="Acquisition purpose by cluster"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        if "country" in filtered:
            geo = filtered["country"].fillna("Missing").value_counts().head(12).reset_index()
            geo.columns = ["Country", "Clients"]
            st.plotly_chart(px.bar(geo, x="Country", y="Clients", title="Geographic buyer distribution (country)"), use_container_width=True)
    with right:
        if "loan_applied" in filtered:
            loan = pd.crosstab(filtered["cluster"], filtered["loan_applied"].fillna("Missing")).reset_index()
            loan_long = loan.melt(id_vars="cluster", var_name="Loan applied", value_name="Clients")
            st.plotly_chart(px.bar(loan_long, x="cluster", y="Clients", color="Loan applied",
                                   barmode="stack", title="Financing pattern by cluster"), use_container_width=True)

    numeric_cols = [c for c in ["age", "satisfaction_score", "property_record_count", "mean_property_price", "mean_floor_area_sqft"] if c in filtered.columns]
    if numeric_cols:
        st.subheader("Numeric profile by cluster")
        numeric_profile = filtered.groupby("cluster")[numeric_cols].mean().round(2).reset_index()
        st.dataframe(numeric_profile, use_container_width=True, hide_index=True)

    st.subheader("Filtered client records")
    display_cols = [c for c in ["client_id", "cluster", "client_type", "country", "region",
                                "acquisition_purpose", "loan_applied", "satisfaction_score",
                                "property_record_count"] if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
    st.download_button(
        "Export filtered dashboard data",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_buyer_dashboard.csv",
        mime="text/csv",
    )
