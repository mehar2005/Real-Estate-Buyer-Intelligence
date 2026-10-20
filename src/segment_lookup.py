import pandas as pd
import streamlit as st
import plotly.express as px

def _mode(series):
    m = series.dropna().mode()
    return str(m.iloc[0]) if not m.empty else "Not available"

def make_segment_lookup(clustered):
    rows = []
    for cid, group in clustered.groupby("cluster", sort=True):
        row = {
            "cluster": str(cid),
            "client_count": len(group),
            "share_pct": round(len(group) / len(clustered) * 100, 2),
            "dominant_client_type": _mode(group["client_type"]) if "client_type" in group else "Not available",
            "dominant_acquisition_purpose": _mode(group["acquisition_purpose"]) if "acquisition_purpose" in group else "Not available",
            "dominant_loan_applied": _mode(group["loan_applied"]) if "loan_applied" in group else "Not available",
            "dominant_country": _mode(group["country"]) if "country" in group else "Not available",
            "dominant_region": _mode(group["region"]) if "region" in group else "Not available",
            "mean_satisfaction": round(float(group["satisfaction_score"].mean()), 2)
                if "satisfaction_score" in group and group["satisfaction_score"].notna().any() else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)

def render_segment_lookup(result):
    st.header("Buyer Segment Lookup")
    st.write("Explore a segment by cluster identifier and view its profile. This section supports analyst lookup and is not a trained predictive classifier for new clients.")

    clustered = result["clustered"].copy()
    lookup = make_segment_lookup(clustered)
    segment = st.selectbox("Choose a cluster", lookup["cluster"].tolist(), key="lookup_cluster")
    row = lookup.loc[lookup["cluster"] == segment].iloc[0]
    group = clustered.loc[clustered["cluster"].astype(str) == str(segment)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Clients in segment", f"{int(row['client_count']):,}")
    c2.metric("Share of all clients", f"{row['share_pct']:.2f}%")
    c3.metric("Mean satisfaction", f"{row['mean_satisfaction']:.2f}" if pd.notna(row["mean_satisfaction"]) else "N/A")

    st.subheader(f"Cluster {segment} profile")
    st.dataframe(pd.DataFrame({
        "Attribute": ["Dominant client type", "Dominant acquisition purpose", "Dominant loan response",
                      "Dominant country", "Dominant region"],
        "Value": [row["dominant_client_type"], row["dominant_acquisition_purpose"],
                  row["dominant_loan_applied"], row["dominant_country"], row["dominant_region"]]
    }), use_container_width=True, hide_index=True)

    categorical = [c for c in ["client_type", "acquisition_purpose", "loan_applied", "country", "region", "referral_channel"] if c in group]
    if categorical:
        field = st.selectbox("Break down this segment by", categorical, key="lookup_breakdown")
        counts = group[field].fillna("Missing").astype(str).value_counts().reset_index()
        counts.columns = [field, "Clients"]
        st.plotly_chart(px.bar(counts, x=field, y="Clients", title=f"Cluster {segment}: {field.replace('_',' ').title()}"), use_container_width=True)

    id_col = "client_id" if "client_id" in group else None
    visible_cols = [c for c in [id_col, "client_type", "country", "region", "acquisition_purpose",
                                "loan_applied", "satisfaction_score", "cluster"] if c and c in group.columns]
    st.subheader("Client records in selected segment")
    st.dataframe(group[visible_cols], use_container_width=True, hide_index=True)
    st.download_button("Download selected segment clients",
                       group.to_csv(index=False).encode("utf-8"),
                       file_name=f"cluster_{segment}_clients.csv", mime="text/csv")

    st.caption("Cluster IDs are model-generated labels. They may change when the data or preprocessing changes; store model artifacts and version the dataset before using this in production.")
