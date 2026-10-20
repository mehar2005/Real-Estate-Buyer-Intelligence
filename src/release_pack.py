import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st


def _file_record(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def build_release_manifest(clients, properties, result, source_note, project_root=None):
    root = Path(project_root or Path(__file__).resolve().parents[1])
    files = []
    for relative in ["app.py", "requirements.txt", "README.md", "src/data_loader.py", "src/data_cleaning.py", "src/feature_engineering.py", "src/clustering.py"]:
        path = root / relative
        if path.exists():
            files.append(_file_record(path))
    scores = result.get("scores") if result and "error" not in result else None
    selected_score = None
    if scores is not None and not scores.empty:
        rows = scores.loc[scores["Clusters (k)"].astype(int) == int(result["chosen_k"])]
        if not rows.empty and pd.notna(rows.iloc[0]["Silhouette score"]):
            selected_score = float(rows.iloc[0]["Silhouette score"])
    return {
        "project": "Real Estate Buyer Intelligence",
        "project_title": "Machine learning-based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": str(source_note),
        "data_summary": {"client_rows": int(len(clients)), "client_columns": int(len(clients.columns)),
                         "property_rows": int(len(properties)), "property_columns": int(len(properties.columns))},
        "model_summary": {"algorithm": "K-Means", "selected_k": int(result["chosen_k"]) if result and "error" not in result else None,
                          "silhouette_score": selected_score, "modelled_clients": int(result["n_rows"]) if result and "error" not in result else 0,
                          "encoded_features": int(result["n_features"]) if result and "error" not in result else 0},
        "files": files,
        "known_limitations": [
            "Clustering is unsupervised and does not establish ground-truth buyer categories.",
            "No explicit income field is available; wealth tiers should not be inferred.",
            "Property aggregates may not represent verified investment totals.",
            "Scenario planner outputs depend on user-entered assumptions.",
            "A manifest records metadata and hashes; it is not a security audit or deployment certificate."
        ]
    }


def render_release_pack(clients, properties, result, source_note):
    st.header("Reproducibility & Release Pack")
    st.write("Create a compact release manifest describing the active datasets, clustering run, key source-file hashes, and known limitations.")
    manifest = build_release_manifest(clients, properties, result, source_note)
    model = manifest["model_summary"]
    a, b, c = st.columns(3)
    a.metric("Client records", f"{manifest['data_summary']['client_rows']:,}")
    b.metric("Property records", f"{manifest['data_summary']['property_rows']:,}")
    c.metric("Selected cluster count", str(model["selected_k"] or "Unavailable"))
    st.subheader("Release manifest")
    st.json(manifest, expanded=False)
    st.download_button("Download release_manifest.json", json.dumps(manifest, indent=2).encode("utf-8"),
                       file_name="release_manifest.json", mime="application/json")
    file_table = pd.DataFrame(manifest["files"])
    if not file_table.empty:
        st.subheader("Tracked source files")
        st.dataframe(file_table, use_container_width=True, hide_index=True)
        st.download_button("Download file integrity table", file_table.to_csv(index=False).encode("utf-8"),
                           file_name="release_file_hashes.csv", mime="text/csv")
    st.subheader("Before release")
    checks = [
        "Run the app in a clean virtual environment and verify every page.",
        "Confirm dataset permissions and remove any sensitive fields before sharing.",
        "Review segment descriptions and business assumptions with a domain mentor.",
        "Keep a copy of the exact datasets and manifest used for the submitted results.",
        "Test downloads and document any known limitations in the research paper."
    ]
    for item in checks:
        st.checkbox(item, key="release_" + str(abs(hash(item))))
    st.caption("The manifest is generated from the current project files and active run. File hashes help detect later changes; they do not prove data quality or model validity.")
