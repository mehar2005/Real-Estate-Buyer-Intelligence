from datetime import datetime, timezone
import html
import pandas as pd
import streamlit as st

def _build_summary(result, source_note):
    clustered = result["clustered"]
    scores = result["scores"]
    selected = int(result["chosen_k"])
    silhouette = scores.loc[scores["Clusters (k)"] == selected, "Silhouette score"]
    silhouette_value = float(silhouette.iloc[0]) if not silhouette.empty and pd.notna(silhouette.iloc[0]) else None
    counts = clustered["cluster"].value_counts().sort_index()
    lines = [
        "# Buyer Segmentation and Investment Profiling — Results Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Data source: {source_note}",
        "",
        "## Method",
        "Client-level features were prepared from the client table and linked property activity. Numeric variables were imputed with medians and standardized. Categorical variables were imputed with the most frequent category and one-hot encoded. K-Means candidates were evaluated using inertia (Elbow Method) and silhouette score.",
        "",
        "## Results",
        f"- Number of client records modelled: {len(clustered):,}",
        f"- Encoded feature count: {result['n_features']:,}",
        f"- Selected candidate cluster count: {selected}",
        f"- Silhouette score for selected candidate: {silhouette_value:.4f}" if silhouette_value is not None else "- Silhouette score for selected candidate: unavailable",
        "",
        "### Cluster distribution",
    ]
    for cluster, count in counts.items():
        lines.append(f"- Cluster {cluster}: {int(count):,} clients ({count / len(clustered) * 100:.2f}%)")
    lines.extend([
        "",
        "## Interpretation cautions",
        "- Cluster IDs are arbitrary and do not have an inherent business meaning.",
        "- Segment descriptions should be derived from cluster profiles and validated by domain stakeholders.",
        "- The client dataset does not contain an explicit income field; do not infer high-income status.",
        "- Aggregated property prices are based on linked records and should not be treated as verified client investment totals.",
        "- Results describe associations in the available dataset and do not establish causation.",
        "",
        "## Suggested next steps",
        "- Validate segment profiles with business stakeholders.",
        "- Run controlled marketing tests and compare measurable outcomes such as qualified enquiries or conversions.",
        "- Collect additional validated fields if investment capacity or property portfolio value must be analysed.",
    ])
    return "\n".join(lines)

def render_report_exports(result, source_note):
    st.header("Section 8 — Reports & Exports")
    st.write("Export the analysis outputs for the internship submission and for further review outside the Streamlit application.")

    clustered = result["clustered"].copy()
    profile_rows = []
    for cluster, group in clustered.groupby("cluster", sort=True):
        row = {"cluster": cluster, "client_count": len(group),
               "share_pct": round(len(group) / len(clustered) * 100, 2)}
        for col in ["age", "satisfaction_score", "property_record_count", "sold_listing_count",
                    "mean_property_price", "mean_floor_area_sqft"]:
            if col in group.columns:
                row[f"mean_{col}"] = round(float(group[col].mean()), 3) if group[col].notna().any() else None
        for col in ["client_type", "acquisition_purpose", "loan_applied", "country", "region"]:
            if col in group.columns:
                mode = group[col].dropna().mode()
                row[f"top_{col}"] = str(mode.iloc[0]) if not mode.empty else "Not available"
        profile_rows.append(row)
    profile = pd.DataFrame(profile_rows)
    summary = _build_summary(result, source_note)
    metrics = {
        "metric": ["clients_modelled", "encoded_features", "selected_cluster_count", "silhouette_score"],
        "value": [len(clustered), result["n_features"], result["chosen_k"],
                  round(float(result["scores"].loc[result["scores"]["Clusters (k)"] == result["chosen_k"], "Silhouette score"].iloc[0]), 5)
                  if result["scores"].loc[result["scores"]["Clusters (k)"] == result["chosen_k"], "Silhouette score"].notna().any()
                  else None]
    }
    scores = result["scores"].copy()

    a, b, c = st.columns(3)
    a.metric("Exportable records", f"{len(clustered):,}")
    b.metric("Cluster profiles", len(profile))
    c.metric("Candidate k values", len(scores))

    st.subheader("Research-paper summary")
    st.markdown(summary)
    st.download_button(
        "Download results summary (Markdown)",
        summary.encode("utf-8"),
        file_name="buyer_segmentation_results.md",
        mime="text/markdown",
    )

    st.subheader("Download tables")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Clustered client dataset (CSV)", clustered.to_csv(index=False).encode("utf-8"),
                           file_name="buyer_segments.csv", mime="text/csv")
        st.download_button("Cluster profiles (CSV)", profile.to_csv(index=False).encode("utf-8"),
                           file_name="cluster_profiles.csv", mime="text/csv")
    with col2:
        st.download_button("Model selection scores (CSV)", scores.to_csv(index=False).encode("utf-8"),
                           file_name="cluster_model_scores.csv", mime="text/csv")
        st.download_button("Summary metrics (CSV)", pd.DataFrame(metrics).to_csv(index=False).encode("utf-8"),
                           file_name="model_summary_metrics.csv", mime="text/csv")

    # Excel workbook assembled in memory for convenient submission.
    from io import BytesIO
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        clustered.to_excel(writer, sheet_name="Buyer Segments", index=False)
        profile.to_excel(writer, sheet_name="Cluster Profiles", index=False)
        scores.to_excel(writer, sheet_name="Model Scores", index=False)
        pd.DataFrame(metrics).to_excel(writer, sheet_name="Summary Metrics", index=False)
    st.download_button(
        "Download all results as Excel workbook",
        excel_buffer.getvalue(),
        file_name="buyer_segmentation_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    with st.expander("Submission checklist"):
        st.markdown("""
        - [ ] Include the project problem statement and objectives.
        - [ ] Describe both datasets, their columns, and the client-property join key.
        - [ ] Include EDA figures with captions.
        - [ ] Document cleaning, feature engineering, encoding, and scaling.
        - [ ] Report Elbow and silhouette results and explain why the selected k was chosen.
        - [ ] Include cluster profiles, limitations, and practical recommendations.
        - [ ] Avoid claiming causality or unsupported income/wealth categories.
        """)
