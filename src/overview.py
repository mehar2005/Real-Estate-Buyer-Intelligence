import streamlit as st
import pandas as pd

def render_project_overview(clients, properties, combined, source_note):
    st.header("Project Overview")
    st.write(
        "This application analyses real-estate buyer profiles and property activity "
        "to identify meaningful buyer segments and investment patterns."
    )
    st.info(f"Data source: {source_note}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Client records", f"{len(clients):,}" if clients is not None else "—")
    c2.metric("Property listings", f"{len(properties):,}" if properties is not None else "—")
    c3.metric("Matched property-client rows", f"{int((combined['_merge'] == 'both').sum()):,}" if combined is not None and "_merge" in combined else "—")

    st.subheader("Business problem")
    st.write(
        "The project brief states that buyer behavior varies by customer type, geography, "
        "investment purpose, and financing behavior. Without segmentation, marketing can "
        "be inefficient and recommendations too generic."
    )
    st.subheader("Project objectives")
    st.markdown("""
    - Prepare and validate client and property datasets.
    - Profile buyer demographics, acquisition purpose, financing, and satisfaction.
    - Build buyer segments using clustering methods such as K-Means and hierarchical clustering.
    - Compare candidate cluster counts with the Elbow Method and Silhouette Score.
    - Present interpretable insights in an interactive Streamlit dashboard.
    """)
    st.subheader("Planned implementation workflow")
    st.markdown("""
    1. Project overview and data ingestion
    2. Data quality assessment and cleaning
    3. Exploratory data analysis
    4. Feature engineering and encoding
    5. Scaling and clustering
    6. Cluster-count evaluation and comparison
    7. Segment profiling and interpretation
    8. Interactive dashboard, exports, and research-paper outputs
    """)
    st.warning(
        "The supplied client dataset does not include an income or investment-amount field. "
        "Therefore, labels such as “high income” or “large investments” must not be claimed "
        "unless supported by additional data. Segments will be named from observed features."
    )

def render_data_quality(raw_clients, raw_properties, clients_clean, properties_clean,
                        clients_report, properties_report, source_note):
    st.header("Data Loading & Quality")
    st.caption(source_note)

    if raw_clients is None and raw_properties is None:
        st.warning("Upload clients.csv and properties.csv, or place them in data/raw/.")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Client dataset")
        if raw_clients is not None:
            st.metric("Rows", f"{len(raw_clients):,}")
            st.metric("Columns", len(raw_clients.columns))
            st.dataframe(clients_report, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Property dataset")
        if raw_properties is not None:
            st.metric("Rows", f"{len(raw_properties):,}")
            st.metric("Columns", len(raw_properties.columns))
            st.dataframe(properties_report, use_container_width=True, hide_index=True)

    if clients_clean is not None and properties_clean is not None:
        st.subheader("Join validation")
        client_ids = set(clients_clean["client_id"].dropna().astype(str)) if "client_id" in clients_clean else set()
        refs = properties_clean["client_ref"].dropna().astype(str) if "client_ref" in properties_clean else pd.Series(dtype=str)
        matched = refs.isin(client_ids)
        a, b, c = st.columns(3)
        a.metric("Property rows with client reference", f"{len(refs):,}")
        b.metric("Matched references", f"{int(matched.sum()):,}")
        c.metric("Unmatched references", f"{int((~matched).sum()):,}")
        st.caption("Unmatched property references are retained for audit and are not silently discarded.")

def render_dataset_preview(df, label):
    st.write(f"Shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"Download cleaned {label}.csv",
        data=csv,
        file_name=f"{label}_cleaned.csv",
        mime="text/csv",
        key=f"download_{label}",
    )
