import streamlit as st
import pandas as pd

from src.data_loader import load_uploaded_or_default_data
from src.data_cleaning import clean_clients, clean_properties, build_customer_property_dataset
from src.overview import render_project_overview, render_data_quality, render_dataset_preview
from src.eda import render_eda
from src.feature_engineering import build_model_matrix, build_client_features, render_feature_engineering
from src.clustering import run_clustering, render_clustering
from src.segment_insights import render_segment_interpretation
from src.dashboard import render_analytics_dashboard
from src.reporting import render_report_exports
from src.validation import render_model_validation
from src.explainability import render_cluster_explainability
from src.recommendations import render_business_recommendations
from src.segment_lookup import render_segment_lookup
from src.deployment import render_deployment_readiness
from src.monitoring import render_data_monitoring
from src.comparison import render_cluster_comparison
from src.documentation import render_technical_documentation
from src.submission import render_submission_readiness
from src.scenario_planner import render_scenario_planner
from src.cohort_explorer import render_cohort_explorer
from src.release_pack import render_release_pack

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background: rgba(128,128,128,0.08); padding: 13px; border-radius: 12px;}
    div[data-testid="stExpander"] {border-radius: 10px;}
    h1, h2, h3 {letter-spacing: -0.02em;}
</style>
""", unsafe_allow_html=True)

st.title("🏙️ Real Estate Buyer Intelligence")
st.caption("Machine learning-based buyer segmentation and investment profiling")

with st.sidebar:
    st.header("Navigation")
    section = st.radio(
        "Project sections",
        [
            "1. Project Overview",
            "2. Data Loading & Quality",
            "3. Exploratory Data Analysis",
            "4. Feature Engineering",
            "5. Buyer Clustering",
            "6. Segment Interpretation",
            "7. Analytics Dashboard",
            "8. Reports & Exports",
            "9. Model Validation",
            "10. Cluster Explainability",
            "11. Business Recommendations",
            "12. Buyer Segment Lookup",
            "13. Deployment Readiness",
            "14. Data Monitoring & Drift",
            "15. Clustering Algorithm Comparison",
            "16. Technical Documentation",
            "17. Final Submission Readiness",
            "18. Segment Campaign Scenario Planner",
            "19. Interactive Buyer Cohort Explorer",
            "20. Reproducibility & Release Pack",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.subheader("Data sources")
    clients_file = st.file_uploader("Upload clients.csv", type=["csv"], key="clients_upload")
    properties_file = st.file_uploader("Upload properties.csv", type=["csv"], key="properties_upload")
    st.caption("Upload both files to replace the bundled datasets.")

clients, properties, source_note = load_uploaded_or_default_data(clients_file, properties_file)
if clients is None or properties is None:
    st.warning("Both client and property datasets are needed for the full workflow.")
    st.caption(f"Current data source: {source_note}")
    st.stop()

clients_clean, clients_report = clean_clients(clients)
properties_clean, properties_report = clean_properties(properties)
combined = build_customer_property_dataset(clients_clean, properties_clean)
client_features, feature_notes = build_client_features(clients_clean, properties_clean)
model_matrix, preprocessor, feature_metadata = build_model_matrix(client_features)

# Cluster once per Streamlit rerun, then share the same assignments across sections.
@st.cache_data(show_spinner="Fitting clustering models…")
def cached_clustering(matrix, features, metadata):
    return run_clustering(matrix, features, metadata)

result = None
if model_matrix is not None:
    result = cached_clustering(model_matrix, client_features, feature_metadata)

if section == "Project Overview":
    render_project_overview(clients, properties, combined, source_note)
elif section == "Data Loading & Quality":
    render_data_quality(
        clients, properties, clients_clean, properties_clean,
        clients_report, properties_report, source_note
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cleaned clients preview")
        render_dataset_preview(clients_clean, "clients")
    with c2:
        st.subheader("Cleaned properties preview")
        render_dataset_preview(properties_clean, "properties")
elif section == "Exploratory Data Analysis":
    render_eda(clients_clean, properties_clean, client_features)
elif section == "Feature Engineering":
    render_feature_engineering(client_features, feature_notes)
elif section == "Buyer Clustering":
    if result is None:
        st.error("The available data does not contain enough usable features for clustering.")
    else:
        render_clustering(result, client_features, model_matrix, feature_metadata)
elif section == "Segment Interpretation":
    if result is None or "error" in result:
        st.error(result.get("error", "Clustering could not be run.") if result else "Clustering could not be run.")
    else:
        render_segment_interpretation(result)
elif section == "Analytics Dashboard":
    if result is None or "error" in result:
        st.error(result.get("error", "Clustering could not be run.") if result else "Clustering could not be run.")
    else:
        render_analytics_dashboard(result)
elif section == "Reports & Exports":
    if result is None or "error" in result:
        st.error(result.get("error", "Clustering could not be run.") if result else "Clustering could not be run.")
    else:
        render_report_exports(result, source_note)
elif section == "Model Validation":
    if result is None or "error" in result:
        st.error(result.get("error", "Clustering could not be run.") if result else "Clustering could not be run.")
    else:
        render_model_validation(result, model_matrix)
elif section == "1Cluster Explainability":
    if result is None or "error" in result:
        st.error(result.get("error", "Clustering could not be run.") if result else "Clustering could not be run.")
    else:
        result["preprocessor"] = preprocessor
        render_cluster_explainability(result, model_matrix, feature_metadata)
elif section == "1Business Recommendations":
    if result is None or "error" in result:
        st.error(result.get("error", "Clustering could not be run.") if result else "Clustering could not be run.")
    else:
        render_business_recommendations(result)
elif section == "1Buyer Segment Lookup":
    if result is None or "error" in result:
        st.error(result.get("error", "Clustering could not be run.") if result else "Clustering could not be run.")
    else:
        render_segment_lookup(result)
elif section == "1Deployment Readiness":
    if result is None or "error" in result:
        st.error(result.get("error", "Clustering could not be run.") if result else "Clustering could not be run.")
    else:
        render_deployment_readiness(result, preprocessor, feature_metadata, source_note)
elif section == "1Data Monitoring & Drift":
    render_data_monitoring(clients_clean, properties_clean)
elif section == "1Clustering Algorithm Comparison":
    if result is None or "error" in result:
        st.error(result.get("error", "Clustering could not be run.") if result else "Clustering could not be run.")
    else:
        render_cluster_comparison(result, model_matrix)
elif section == "1Technical Documentation":
    if result is None or "error" in result:
        st.error(result.get("error", "Clustering could not be run.") if result else "Clustering could not be run.")
    else:
        render_technical_documentation(result, source_note, clients, properties, feature_metadata)
elif section == "1Final Submission Readiness":
    render_submission_readiness()
elif section == "1Interactive Buyer Cohort Explorer":
    if result is None or "error" in result:
        st.error(result.get("error", "Clustering could not be run.") if result else "Clustering could not be run.")
    else:
        render_cohort_explorer(result)
elif section == "2Reproducibility & Release Pack":
    render_release_pack(clients_clean, properties_clean, result, source_note)

with st.sidebar:
    st.divider()
    st.caption("Implemented: Sections 1–20")
    st.caption(f"Clients: {len(clients_clean):,} · Properties: {len(properties_clean):,}")
