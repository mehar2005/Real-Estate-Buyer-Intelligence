from datetime import datetime, timezone
import pandas as pd
import streamlit as st

def build_methodology_document(result, source_note, client_shape, property_shape, feature_metadata):
    score_rows = result["scores"]
    selected_k = int(result["chosen_k"])
    selected = score_rows.loc[score_rows["Clusters (k)"] == selected_k]
    silhouette = float(selected["Silhouette score"].iloc[0]) if not selected.empty and pd.notna(selected["Silhouette score"].iloc[0]) else None
    numeric = feature_metadata.get("numeric", [])
    categorical = feature_metadata.get("categorical", [])
    text = f"""# Technical Methodology — Real Estate Buyer Intelligence

Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## 1. Project objective
Identify groups of real-estate clients with similar observable demographic, behavioral, and linked property-activity characteristics. Use these groups to support buyer profiling and generate testable marketing hypotheses.

## 2. Data sources
- Source note: {source_note}
- Client table shape before cleaning: {client_shape[0]} rows × {client_shape[1]} columns
- Property table shape before cleaning: {property_shape[0]} rows × {property_shape[1]} columns
- Join key: `properties.client_ref` to `clients.client_id`

## 3. Data preparation
- Text fields are stripped of surrounding whitespace and common category labels are standardized.
- Duplicate client IDs and listing IDs are removed by keeping the first record.
- Date fields are parsed with invalid values coerced to missing.
- Satisfaction scores outside the supported 1–5 range are treated as missing.
- Invalid non-positive property prices and floor areas are treated as missing.
- Property activity is aggregated to the client level using the available client reference.

## 4. Feature engineering and preprocessing
- Numeric features: {", ".join(numeric) if numeric else "No numeric features selected"}.
- Categorical features: {", ".join(categorical) if categorical else "No categorical features selected"}.
- Numeric missing values are imputed with the median and numeric features are standardized.
- Categorical missing values are imputed with the most frequent category and categories are one-hot encoded.
- Direct identifiers, names, and raw date of birth are excluded from the clustering input.

## 5. Clustering and model selection
- Primary algorithm: K-Means.
- Candidate cluster counts: 2 through 10, bounded by the number of available records.
- Selection criterion: highest silhouette score among the tested candidates.
- Elbow/inertia curve is also displayed as a secondary diagnostic.
- Selected candidate cluster count: {selected_k}.
- Selected silhouette score: {f"{silhouette:.4f}" if silhouette is not None else "Unavailable"}.
- Hierarchical clustering is used as a comparison when feasible.

## 6. Outputs
- Cluster assignments per client.
- Cluster profiles and distribution.
- EDA charts and missing-value summaries.
- Model selection scores and downloadable report tables.
- Segment-specific recommendations framed as experiments to validate.

## 7. Limitations
- Clustering is unsupervised; cluster IDs have no inherent meaning.
- The source client data has no explicit income field.
- Property price aggregates represent linked property records, not necessarily verified client investment totals.
- Internal cluster metrics do not prove commercial value or causality.
- Recommendations require stakeholder review and outcome-based testing.

## 8. Reproducibility
Use the same source data, dependency versions, preprocessing, and random seed when comparing results. Version the data and model artifacts together.
"""
    return text

def render_technical_documentation(result, source_note, clients, properties, feature_metadata):
    st.header("Section 16 — Technical Documentation")
    st.write("Generate a methodology document for the project report and inspect the current modeling configuration.")
    text = build_methodology_document(result, source_note, clients.shape, properties.shape, feature_metadata)
    st.markdown(text)
    st.download_button("Download technical methodology (Markdown)", text.encode("utf-8"),
                       file_name="technical_methodology.md", mime="text/markdown")
    st.subheader("Model selection table")
    st.dataframe(result["scores"].round(4), use_container_width=True, hide_index=True)
    st.download_button("Download model selection table (CSV)",
                       result["scores"].to_csv(index=False).encode("utf-8"),
                       file_name="model_selection_scores.csv", mime="text/csv")
