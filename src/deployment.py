from io import BytesIO
import json
import platform
import sys
import pandas as pd
import streamlit as st
import sklearn
import numpy as np

def render_deployment_readiness(result, preprocessor, feature_metadata, source_note):
    st.header("Section 13 — Deployment Readiness & Model Packaging")
    st.write("Prepare a reproducible model bundle and review practical deployment requirements. This page packages the preprocessing object and clustering estimator for later reuse; it does not publish the app to a public hosting service.")

    clustered = result["clustered"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Clients in current model", f"{len(clustered):,}")
    c2.metric("Chosen clusters", result["chosen_k"])
    c3.metric("Encoded features", result["n_features"])

    st.subheader("Runtime environment")
    env = pd.DataFrame([
        ["Python", sys.version.split()[0]],
        ["Platform", platform.platform()],
        ["Pandas", pd.__version__],
        ["NumPy", np.__version__],
        ["Scikit-learn", sklearn.__version__],
        ["Data source", source_note],
    ], columns=["Component", "Value"])
    st.dataframe(env, use_container_width=True, hide_index=True)

    st.subheader("Model artifact bundle")
    st.warning("Pickle/joblib model artifacts can execute code when loaded. Only load artifacts from trusted sources and keep versions aligned.")
    if st.button("Prepare model bundle"):
        try:
            import joblib
            from src.feature_engineering import build_model_matrix
            from src.clustering import KMeans
            bundle = {
                "preprocessor": preprocessor,
                "kmeans_model": result["kmeans"],
                "chosen_k": int(result["chosen_k"]),
                "feature_metadata": feature_metadata,
                "feature_names": list(preprocessor.get_feature_names_out()),
                "training_rows": int(len(clustered)),
                "training_features": int(result["n_features"]),
                "source_note": source_note,
                "sklearn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
            }
            buffer = BytesIO()
            joblib.dump(bundle, buffer)
            st.download_button(
                "Download trusted model bundle (.joblib)",
                buffer.getvalue(),
                file_name="buyer_segmentation_model.joblib",
                mime="application/octet-stream",
            )
            metadata = {
                "chosen_k": int(result["chosen_k"]),
                "training_rows": int(len(clustered)),
                "training_features": int(result["n_features"]),
                "feature_metadata": feature_metadata,
                "source_note": source_note,
                "python_version": sys.version.split()[0],
                "sklearn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
            }
            st.download_button(
                "Download model metadata (JSON)",
                json.dumps(metadata, indent=2).encode("utf-8"),
                file_name="buyer_segmentation_model_metadata.json",
                mime="application/json",
            )
            st.success("Model bundle and metadata are ready to download.")
        except Exception as exc:
            st.error(f"Could not prepare model bundle: {exc}")

    st.subheader("Deployment checklist")
    checklist = [
        "Pin dependency versions and test from a clean environment.",
        "Store preprocessing and clustering artifacts together.",
        "Version the training data and model metadata.",
        "Add input-schema and missing-field validation for future uploads.",
        "Protect uploaded client data and avoid committing private datasets to public repositories.",
        "Use trusted artifact storage and access controls.",
        "Monitor cluster sizes and feature distributions after data updates.",
        "Document retraining criteria and rollback steps.",
    ]
    for item in checklist:
        st.checkbox(item, key="deploy_" + str(abs(hash(item))))

    st.info("For a Streamlit-hosted deployment, use a private repository or appropriate secrets/data-access controls if the data is confidential. Do not put API keys or database passwords in source code.")
