import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

def run_clustering(matrix, features, feature_metadata):
    n = len(features)
    if n < 3 or matrix.shape[1] == 0:
        return {"error": "At least 3 client records and one usable feature are required."}

    # Use a sample for silhouette computation when datasets become large.
    max_k = min(10, n - 1)
    k_values = list(range(2, max_k + 1))
    inertias, silhouettes = [], []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(matrix)
        inertias.append(float(model.inertia_))
        try:
            silhouettes.append(float(silhouette_score(
                matrix, labels,
                sample_size=min(2000, n),
                random_state=42
            )))
        except Exception:
            silhouettes.append(float("nan"))

    scores = pd.DataFrame({
        "Clusters (k)": k_values,
        "Inertia": inertias,
        "Silhouette score": silhouettes,
    })
    finite_scores = scores.dropna(subset=["Silhouette score"])
    if finite_scores.empty:
        best_k = min(4, max_k)
    else:
        best_k = int(finite_scores.loc[finite_scores["Silhouette score"].idxmax(), "Clusters (k)"])

    chosen_k = int(best_k)
    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(matrix)
    result_features = features.copy()
    result_features["cluster"] = labels.astype(int).astype(str)

    # Hierarchical clustering is included as a comparison model.
    hierarchical_labels = None
    if n <= 5000:
        try:
            hierarchical_labels = AgglomerativeClustering(n_clusters=chosen_k, linkage="ward").fit_predict(matrix)
        except Exception:
            hierarchical_labels = None

    return {
        "scores": scores,
        "best_k": best_k,
        "chosen_k": chosen_k,
        "labels": labels,
        "kmeans": kmeans,
        "clustered": result_features,
        "hierarchical_labels": hierarchical_labels,
        "n_rows": n,
        "n_features": matrix.shape[1],
        "feature_metadata": feature_metadata,
    }

def _segment_name(group):
    # Data-grounded descriptive labels only; do not infer wealth or investment value.
    purpose = group.get("acquisition_purpose", pd.Series(dtype=object)).mode()
    client_type = group.get("client_type", pd.Series(dtype=object)).mode()
    loan = group.get("loan_applied", pd.Series(dtype=object)).mode()
    if len(client_type) and str(client_type.iloc[0]).lower() == "corporate":
        return "Corporate clients"
    if len(purpose) and "invest" in str(purpose.iloc[0]).lower():
        return "Investment-purpose buyers"
    if len(loan) and str(loan.iloc[0]).lower() == "yes":
        return "Loan-applying buyers"
    if len(purpose):
        return f"{str(purpose.iloc[0])} buyers"
    return "Buyer segment"

def _profile_clusters(clustered):
    numeric = clustered.select_dtypes(include=["number"]).columns.tolist()
    summary = clustered.groupby("cluster").size().rename("client_count").to_frame()
    summary["share_pct"] = (summary["client_count"] / len(clustered) * 100).round(2)
    for col in numeric:
        summary[f"avg_{col}"] = clustered.groupby("cluster")[col].mean().round(2)
    for col in ["client_type", "acquisition_purpose", "loan_applied", "country", "region", "referral_channel"]:
        if col in clustered.columns:
            summary[f"top_{col}"] = clustered.groupby("cluster")[col].agg(
                lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"
            )
    return summary.reset_index()

def render_clustering(result, features, matrix, feature_metadata):
    st.header("Buyer Clustering")
    st.write("This section compares K-Means clustering across candidate cluster counts, selects a candidate using silhouette score, and profiles the resulting groups.")
    if "error" in result:
        st.error(result["error"])
        return

    scores = result["scores"]
    best_k = result["best_k"]
    clustered = result["clustered"]
    selected_score = scores.loc[scores["Clusters (k)"] == best_k, "Silhouette score"].iloc[0]

    a, b, c, d = st.columns(4)
    a.metric("Clients modelled", f"{result['n_rows']:,}")
    b.metric("Features after encoding", f"{result['n_features']:,}")
    c.metric("Suggested cluster count", best_k)
    d.metric("Silhouette score", f"{selected_score:.3f}" if pd.notna(selected_score) else "N/A")

    st.subheader("Optimal cluster selection")
    left, right = st.columns(2)
    with left:
        fig = px.line(scores, x="Clusters (k)", y="Inertia", markers=True, title="Elbow method")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(scores, x="Clusters (k)", y="Silhouette score", markers=True,
                      title="Silhouette score by cluster count")
        fig.add_vline(x=best_k, line_dash="dash", annotation_text=f"Selected k={best_k}")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(scores.round(4), use_container_width=True, hide_index=True)
    st.caption("The selected k is the candidate with the highest computed silhouette score. The Elbow curve should also be reviewed; clustering quality is not proof of business value.")

    st.subheader("Buyer segment distribution")
    counts = clustered["cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Clients")
    counts["Share (%)"] = (counts["Clients"] / len(clustered) * 100).round(2)
    left, right = st.columns(2)
    with left:
        fig = px.bar(counts, x="Cluster", y="Clients", text="Clients", title="Clients per cluster")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.pie(counts, names="Cluster", values="Clients", hole=0.45, title="Cluster share")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster profiles")
    profile = _profile_clusters(clustered)
    profile["Suggested descriptive label"] = profile["cluster"].apply(
        lambda cid: _segment_name(clustered.loc[clustered["cluster"] == cid])
    )
    st.dataframe(profile, use_container_width=True, hide_index=True)
    st.download_button(
        "Download clustered client data",
        clustered.to_csv(index=False).encode("utf-8"),
        file_name="buyer_segments.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download cluster profile table",
        profile.to_csv(index=False).encode("utf-8"),
        file_name="cluster_profiles.csv",
        mime="text/csv",
    )

    with st.expander("Compare K-Means with hierarchical clustering"):
        if result["hierarchical_labels"] is None:
            st.info("Hierarchical clustering comparison was skipped because the dataset was too large or the algorithm could not be fitted.")
        else:
            comparison = clustered[["client_id"]].copy() if "client_id" in clustered else pd.DataFrame(index=clustered.index)
            comparison["kmeans_cluster"] = result["labels"]
            comparison["hierarchical_cluster"] = result["hierarchical_labels"]
            agreement = (comparison["kmeans_cluster"] == comparison["hierarchical_cluster"]).mean()
            st.metric("Same numeric label share", f"{agreement:.1%}")
            st.caption("Cluster numbers are arbitrary identifiers, so direct label agreement is only a rough diagnostic; use profile similarity and adjusted Rand index for a rigorous comparison.")
            st.dataframe(pd.crosstab(comparison["kmeans_cluster"], comparison["hierarchical_cluster"]),
                         use_container_width=True)

    with st.expander("Modeling notes and limitations"):
        st.markdown("""
        - K-Means is fitted to standardized numeric and one-hot encoded categorical features.
        - The Elbow Method uses within-cluster sum of squares (inertia); the silhouette score evaluates separation and cohesion.
        - `client_id`, names, and raw date of birth are excluded from the model.
        - Property aggregates describe linked records; they are not necessarily verified client investment totals.
        - The brief's suggested buyer labels are hypotheses, not labels guaranteed by the dataset. Interpret clusters using the displayed profiles.
        """)
