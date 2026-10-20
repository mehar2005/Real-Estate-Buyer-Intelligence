import pandas as pd
import streamlit as st

CHECKLIST = [
    ("Application", "Run `streamlit run app.py` from a clean environment."),
    ("Data", "Confirm both CSV files load and verify row/column counts."),
    ("Cleaning", "Review missing values, duplicates, invalid values, and unmatched client references."),
    ("EDA", "Save key charts and explain the patterns without claiming causality."),
    ("Feature engineering", "Document aggregation, encoding, scaling, and excluded identifiers."),
    ("Clustering", "Include Elbow and silhouette results and justify the selected cluster count."),
    ("Validation", "Report internal metrics, cluster sizes, and algorithm comparison."),
    ("Interpretation", "Describe each segment from actual profile statistics."),
    ("Dashboard", "Test filters, downloads, and empty-filter states."),
    ("Research paper", "Include problem statement, data, methodology, results, limitations, and recommendations."),
    ("Reproducibility", "Include requirements.txt, README, and clear run instructions."),
    ("Privacy", "Remove secrets and avoid publishing private customer data in a public repository."),
]

def render_submission_readiness():
    st.header("Section 17 — Final Submission Readiness")
    st.write("Use this checklist before handing the project to Unified Mentor. Tick items only after you have personally verified them in your local environment.")
    completed = []
    for category, task in CHECKLIST:
        key = "submission_" + category.lower().replace(" ", "_")
        done = st.checkbox(f"**{category}:** {task}", key=key)
        completed.append(done)
    pct = round(sum(completed) / len(completed) * 100)
    st.progress(pct / 100)
    st.metric("Checklist completion", f"{sum(completed)}/{len(completed)} ({pct}%)")
    st.caption("This is a local interactive checklist. It does not certify the project or replace your mentor's required format.")

    st.subheader("Final deliverable inventory")
    inventory = pd.DataFrame([
        ["Streamlit application", "app.py and src modules", "Run and test locally"],
        ["Raw datasets", "data/raw/clients.csv, properties.csv", "Confirm permitted use and privacy"],
        ["Dependencies", "requirements.txt", "Install in a clean environment"],
        ["Documentation", "README.md and technical_methodology.md export", "Check accuracy against final run"],
        ["Model outputs", "CSV / Excel downloads from the app", "Regenerate after final model run"],
        ["Research paper", "Prepared separately using the mentor's required format", "Do not assume the generated Markdown replaces the required paper template"],
    ], columns=["Deliverable", "Location / output", "Final check"])
    st.dataframe(inventory, use_container_width=True, hide_index=True)
    st.download_button("Download submission checklist (CSV)",
                       inventory.to_csv(index=False).encode("utf-8"),
                       file_name="submission_checklist.csv", mime="text/csv")
