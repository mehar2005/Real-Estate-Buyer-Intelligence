import pandas as pd
import streamlit as st
import plotly.express as px

def _mode(series):
    values = series.dropna().mode()
    return str(values.iloc[0]) if not values.empty else "Not available"

def _profile(clustered):
    rows = []
    for cluster, group in clustered.groupby("cluster", sort=True):
        row = {
            "Cluster": str(cluster),
            "Clients": int(len(group)),
            "Share (%)": round(len(group) / len(clustered) * 100, 2),
        }
        for col in ["client_type", "acquisition_purpose", "loan_applied", "country", "region", "referral_channel"]:
            if col in group.columns:
                row[f"Most common {col.replace('_', ' ')}"] = _mode(group[col])
        for col in ["age", "satisfaction_score", "property_record_count", "sold_listing_count",
                    "mean_property_price", "mean_floor_area_sqft"]:
            if col in group.columns and pd.api.types.is_numeric_dtype(group[col]):
                row[f"Average {col.replace('_', ' ')}"] = round(float(group[col].mean()), 2) if group[col].notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows)

def _recommendations(group, cluster):
    recommendations = []
    purpose = _mode(group["acquisition_purpose"]).lower() if "acquisition_purpose" in group else ""
    client_type = _mode(group["client_type"]).lower() if "client_type" in group else ""
    loan = _mode(group["loan_applied"]).lower() if "loan_applied" in group else ""
    satisfaction = group["satisfaction_score"].mean() if "satisfaction_score" in group else None
    property_count = group["property_record_count"].mean() if "property_record_count" in group else None

    if "corporate" in client_type:
        recommendations.append("Test business-focused outreach and portfolio-level property information for this segment.")
    if "invest" in purpose:
        recommendations.append("Prioritize investment-oriented content and compare engagement with relevant property categories.")
    elif purpose:
        recommendations.append("Tailor messaging to the observed acquisition purpose instead of using a generic investor message.")
    if loan in {"yes", "y", "true"}:
        recommendations.append("Provide clear financing information and measure whether financing guidance improves enquiry conversion.")
    if satisfaction is not None and pd.notna(satisfaction) and satisfaction < 3:
        recommendations.append("Review service feedback and investigate friction points before increasing promotional spend.")
    elif satisfaction is not None and pd.notna(satisfaction) and satisfaction >= 4:
        recommendations.append("Consider opt-in feedback or referral experiments with this relatively satisfied group.")
    if property_count is not None and pd.notna(property_count) and property_count > 1:
        recommendations.append("Check whether repeat property activity indicates repeat interest; validate against business records.")
    if not recommendations:
        recommendations.append("Run a small, measurable campaign test for this segment and compare enquiry or conversion outcomes.")
    recommendations.append("Validate this segment with business stakeholders; clustering alone does not prove purchase intent or profitability.")
    return recommendations

def render_segment_interpretation(result):
    st.header("Segment Interpretation & Recommendations")
    st.write("Interpret clusters using observed client and property features. The descriptive labels are generated from available fields; they are not ground-truth buyer categories.")

    clustered = result["clustered"].copy()
    profile = _profile(clustered)
    st.subheader("Segment profile comparison")
    st.dataframe(profile, use_container_width=True, hide_index=True)

    segment_ids = sorted(clustered["cluster"].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else x)
    selected = st.selectbox("Inspect a segment", segment_ids)
    group = clustered.loc[clustered["cluster"].astype(str) == str(selected)]
    st.subheader(f"Cluster {selected}")
    a, b, c = st.columns(3)
    a.metric("Clients", f"{len(group):,}")
    b.metric("Share of clients", f"{len(group) / len(clustered) * 100:.1f}%")
    c.metric("Average satisfaction", f"{group['satisfaction_score'].mean():.2f}" if "satisfaction_score" in group and group["satisfaction_score"].notna().any() else "N/A")

    if "acquisition_purpose" in group:
        purpose_counts = group["acquisition_purpose"].fillna("Missing").value_counts().reset_index()
        purpose_counts.columns = ["Acquisition purpose", "Clients"]
        st.plotly_chart(px.bar(purpose_counts, x="Acquisition purpose", y="Clients",
                               title="Acquisition purpose within selected segment"), use_container_width=True)
    if "country" in group:
        country_counts = group["country"].fillna("Missing").value_counts().head(10).reset_index()
        country_counts.columns = ["Country", "Clients"]
        st.plotly_chart(px.bar(country_counts, x="Country", y="Clients",
                               title="Top countries in selected segment"), use_container_width=True)

    st.subheader("Suggested actions to test")
    for item in _recommendations(group, selected):
        st.markdown(f"- {item}")

    st.warning("The dataset does not provide explicit income or verified investment-portfolio value. Do not name a cluster 'high-income' or 'luxury investor' without additional evidence.")
    st.download_button(
        "Download segment profile",
        profile.to_csv(index=False).encode("utf-8"),
        file_name="segment_interpretation.csv",
        mime="text/csv",
    )
