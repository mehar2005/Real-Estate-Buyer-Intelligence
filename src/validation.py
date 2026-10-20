import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.metrics import confusion_matrix

def _safe_metric(func, matrix, labels):
    try:
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
            return None
        return float(func(matrix, labels))
    except Exception:
        return None

def render_model_validation(result, matrix):
    st.header("Section 9 — Model Validation & Quality Diagnostics")
    st.write("Evaluate the chosen clustering solution using complementary internal metrics and stability checks. These metrics describe geometric cluster quality; they do not establish business usefulness on their own.")

    labels = np.asarray(result["labels"])
    scores = result["scores"]
    k = int(result["chosen_k"])
    silhouette = _safe_metric(silhouette_score, matrix, labels)
    dbi = _safe_metric(davies_bouldin_score, matrix, labels)
    ch = _safe_metric(calinski_harabasz_score, matrix, labels)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Silhouette", f"{silhouette:.3f}" if silhouette is not None else "N/A", help="Higher is generally better; range is -1 to 1.")
    c2.metric("Davies–Bouldin", f"{dbi:.3f}" if dbi is not None else "N/A", help="Lower is generally better.")
    c3.metric("Calinski–Harabasz", f"{ch:,.1f}" if ch is not None else "N/A", help="Higher is generally better for a given dataset.")
    c4.metric("Clusters", k)

    st.subheader("Metric interpretation")
    st.markdown("""
    - **Silhouette score:** combines within-cluster cohesion and separation from other clusters. Values near 1 suggest well-separated groups; values near 0 suggest overlap; negative values can indicate questionable assignments.
    - **Davies–Bouldin index:** compares within-cluster dispersion with between-cluster separation. Lower values are generally preferred.
    - **Calinski–Harabasz index:** compares between-cluster dispersion with within-cluster dispersion. Higher values are generally preferred, but it should be compared across candidate solutions on the same dataset.
    """)

    st.subheader("Repeat-fit stability check")
    seed_a = st.number_input("First random seed", min_value=0, max_value=99999, value=42, step=1)
    seed_b = st.number_input("Second random seed", min_value=0, max_value=99999, value=123, step=1)
    if st.button("Run stability check"):
        model_a = KMeans(n_clusters=k, random_state=int(seed_a), n_init=10).fit(matrix)
        model_b = KMeans(n_clusters=k, random_state=int(seed_b), n_init=10).fit(matrix)
        ari = adjusted_rand_score(model_a.labels_, model_b.labels_)
        st.metric("Adjusted Rand Index (ARI)", f"{ari:.3f}", help="Label-invariant agreement between two partitions; 1 means identical partitions.")
        st.caption("This measures repeat-fit agreement on the same feature matrix. It does not measure whether segments predict future outcomes.")

    if result.get("hierarchical_labels") is not None:
        st.subheader("K-Means vs. hierarchical partition")
        ari = adjusted_rand_score(labels, np.asarray(result["hierarchical_labels"]))
        st.metric("Adjusted Rand Index", f"{ari:.3f}", help="Unlike raw cluster-label matching, ARI is invariant to arbitrary cluster numbers.")
        st.caption("A low ARI means the methods partition the same records differently; inspect the cluster profiles before selecting a business interpretation.")

    st.subheader("Cluster sizes")
    counts = pd.Series(labels).value_counts().sort_index().rename_axis("Cluster").reset_index(name="Records")
    counts["Share (%)"] = (counts["Records"] / len(labels) * 100).round(2)
    st.plotly_chart(px.bar(counts, x="Cluster", y="Records", text="Records", title="Cluster balance"), use_container_width=True)
    st.dataframe(counts, use_container_width=True, hide_index=True)

    st.warning("Internal metrics can favor geometrically separated groups that are not commercially meaningful. Validate segments with domain experts and later business outcomes.")
