import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering

def render_cluster_comparison(result, matrix):
    st.header("Section 15 — Clustering Algorithm Comparison")
    st.write("Compare K-Means and hierarchical clustering across the same preprocessed client feature matrix. Metrics are internal diagnostics, not proof of business value.")

    k = int(result["chosen_k"])
    rows = []
    algorithms = {}
    with st.spinner("Evaluating clustering algorithms..."):
        for name, model in [
            ("K-Means", KMeans(n_clusters=k, random_state=42, n_init=10)),
            ("Hierarchical — Ward", AgglomerativeClustering(n_clusters=k, linkage="ward")),
        ]:
            try:
                labels = model.fit_predict(matrix)
                algorithms[name] = labels
                sil = silhouette_score(matrix, labels, sample_size=min(2000, len(matrix)), random_state=42)
                rows.append({"Algorithm": name, "Clusters": len(np.unique(labels)),
                             "Silhouette score": float(sil),
                             "Smallest cluster": int(pd.Series(labels).value_counts().min()),
                             "Largest cluster": int(pd.Series(labels).value_counts().max())})
            except Exception as exc:
                rows.append({"Algorithm": name, "Clusters": None, "Silhouette score": None,
                             "Smallest cluster": None, "Largest cluster": None, "Error": str(exc)})

    scores = pd.DataFrame(rows)
    st.dataframe(scores.round(4), use_container_width=True, hide_index=True)
    if not scores["Silhouette score"].isna().all():
        st.plotly_chart(px.bar(scores.dropna(subset=["Silhouette score"]),
                               x="Algorithm", y="Silhouette score",
                               title="Silhouette score by clustering algorithm"), use_container_width=True)

    if "K-Means" in algorithms and "Hierarchical — Ward" in algorithms:
        ari = adjusted_rand_score(algorithms["K-Means"], algorithms["Hierarchical — Ward"])
        st.metric("Partition agreement (Adjusted Rand Index)", f"{ari:.3f}")
        counts = pd.crosstab(algorithms["K-Means"], algorithms["Hierarchical — Ward"])
        st.plotly_chart(px.imshow(counts, text_auto=True, aspect="auto",
                                  labels=dict(x="Hierarchical cluster", y="K-Means cluster", color="Clients"),
                                  title="Cluster membership cross-tabulation"), use_container_width=True)
        st.caption("ARI is invariant to arbitrary cluster IDs. A low value indicates the two algorithms group records differently.")

    st.subheader("Choose a model based on evidence")
    st.markdown("""
    - Compare silhouette score alongside cluster balance and interpretability.
    - Review whether clusters are distinguishable using meaningful business variables.
    - Prefer the simpler, more stable model when quality is similar.
    - Document the selection rationale rather than choosing solely by one metric.
    """)
    st.warning("Hierarchical clustering can be memory-intensive on large datasets. If it fails, use K-Means diagnostics and document the limitation.")
