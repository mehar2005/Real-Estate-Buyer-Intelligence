import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

def _numeric_drift(reference, current, col):
    a = pd.to_numeric(reference[col], errors="coerce").dropna()
    b = pd.to_numeric(current[col], errors="coerce").dropna()
    if a.empty or b.empty:
        return {"Feature": col, "Reference mean": None, "Current mean": None, "Mean shift (std units)": None}
    std = float(a.std(ddof=0))
    shift = (float(b.mean()) - float(a.mean())) / std if std > 0 else None
    return {"Feature": col, "Reference mean": round(float(a.mean()), 3),
            "Current mean": round(float(b.mean()), 3),
            "Mean shift (std units)": round(float(shift), 3) if shift is not None else None}

def _categorical_drift(reference, current, col):
    a = reference[col].fillna("Missing").astype(str).value_counts(normalize=True)
    b = current[col].fillna("Missing").astype(str).value_counts(normalize=True)
    cats = sorted(set(a.index) | set(b.index))
    # Total variation distance ranges 0–1; 0 means identical category proportions.
    tvd = 0.5 * sum(abs(float(a.get(cat, 0)) - float(b.get(cat, 0))) for cat in cats)
    return {"Feature": col, "Reference categories": int(reference[col].nunique(dropna=False)),
            "Current categories": int(current[col].nunique(dropna=False)),
            "Total variation distance": round(float(tvd), 4)}

def render_data_monitoring(reference_clients, reference_properties):
    st.header("Data Monitoring & Drift Checks")
    st.write("Compare the bundled/reference datasets with a newly supplied batch to spot changes in numeric distributions and categorical proportions. Drift flags are screening signals, not proof that a model has become invalid.")

    uploaded_clients = st.file_uploader("Upload a new clients batch (CSV)", type=["csv"], key="monitor_clients")
    uploaded_properties = st.file_uploader("Upload a new properties batch (CSV)", type=["csv"], key="monitor_properties")
    if uploaded_clients is None and uploaded_properties is None:
        st.info("Upload at least one new dataset to compare it with the current reference data.")
        return

    if uploaded_clients is not None:
        try:
            current_clients = pd.read_csv(uploaded_clients)
            st.subheader("Client data drift")
            numeric = [c for c in ["satisfaction_score", "age"] if c in reference_clients and c in current_clients]
            categorical = [c for c in ["client_type", "gender", "country", "region", "acquisition_purpose",
                                        "loan_applied", "referral_channel"] if c in reference_clients and c in current_clients]
            rows = [_numeric_drift(reference_clients, current_clients, c) for c in numeric]
            cat_rows = [_categorical_drift(reference_clients, current_clients, c) for c in categorical]
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if cat_rows:
                cat_df = pd.DataFrame(cat_rows)
                st.dataframe(cat_df, use_container_width=True, hide_index=True)
                st.plotly_chart(px.bar(cat_df, x="Feature", y="Total variation distance",
                                       title="Categorical distribution drift (0 = no change)"), use_container_width=True)
            if not rows and not cat_rows:
                st.warning("No overlapping supported client columns were found for drift checks.")
        except Exception as exc:
            st.error(f"Could not read new client batch: {exc}")

    if uploaded_properties is not None:
        try:
            current_properties = pd.read_csv(uploaded_properties)
            st.subheader("Property data drift")
            numeric = [c for c in ["sale_price", "floor_area_sqft"] if c in reference_properties and c in current_properties]
            categorical = [c for c in ["unit_category", "listing_status"] if c in reference_properties and c in current_properties]
            rows = [_numeric_drift(reference_properties, current_properties, c) for c in numeric]
            cat_rows = [_categorical_drift(reference_properties, current_properties, c) for c in categorical]
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if cat_rows:
                cat_df = pd.DataFrame(cat_rows)
                st.dataframe(cat_df, use_container_width=True, hide_index=True)
                st.plotly_chart(px.bar(cat_df, x="Feature", y="Total variation distance",
                                       title="Property category drift (0 = no change)"), use_container_width=True)
            if not rows and not cat_rows:
                st.warning("No overlapping supported property columns were found for drift checks.")
        except Exception as exc:
            st.error(f"Could not read new property batch: {exc}")

    st.caption("Numeric mean shift is standardized by the reference standard deviation. Categorical total variation distance compares category proportions. These simple checks do not replace formal monitoring or retraining validation.")
