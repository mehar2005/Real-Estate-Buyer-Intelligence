import pandas as pd
import streamlit as st

def _mode(series):
    mode = series.dropna().mode()
    return str(mode.iloc[0]) if not mode.empty else "Not available"

def build_recommendation_table(clustered):
    rows = []
    for cid, group in clustered.groupby("cluster", sort=True):
        purpose = _mode(group["acquisition_purpose"]).lower() if "acquisition_purpose" in group else ""
        ctype = _mode(group["client_type"]).lower() if "client_type" in group else ""
        loan = _mode(group["loan_applied"]).lower() if "loan_applied" in group else ""
        satisfaction = group["satisfaction_score"].mean() if "satisfaction_score" in group else None
        size = len(group)
        share = size / len(clustered) * 100
        actions = []
        if "corporate" in ctype:
            actions.append("Test corporate-oriented outreach and multi-property information.")
        if "invest" in purpose:
            actions.append("Test investment-focused property information and follow-up content.")
        elif purpose:
            actions.append(f"Tailor messaging to the observed {purpose} acquisition purpose.")
        if loan in {"yes", "y", "true"}:
            actions.append("Test clear financing guidance and track qualified enquiries.")
        if satisfaction is not None and pd.notna(satisfaction) and satisfaction < 3:
            actions.append("Investigate service friction and feedback before scaling marketing.")
        elif satisfaction is not None and pd.notna(satisfaction) and satisfaction >= 4:
            actions.append("Test an opt-in referral or feedback campaign with this group.")
        if not actions:
            actions.append("Run a small A/B test using segment-specific messages and a measurable outcome.")
        actions.append("Validate recommendations with stakeholders and compare campaign outcomes against a baseline.")
        rows.append({
            "Cluster": str(cid),
            "Client count": size,
            "Share (%)": round(share, 2),
            "Dominant client type": _mode(group["client_type"]) if "client_type" in group else "Not available",
            "Dominant purpose": _mode(group["acquisition_purpose"]) if "acquisition_purpose" in group else "Not available",
            "Dominant loan response": _mode(group["loan_applied"]) if "loan_applied" in group else "Not available",
            "Mean satisfaction": round(float(satisfaction), 2) if satisfaction is not None and pd.notna(satisfaction) else None,
            "Suggested actions": " ".join(actions),
        })
    return pd.DataFrame(rows)

def render_business_recommendations(result):
    st.header("Section 11 — Business Recommendations & Action Plan")
    st.write("Recommendations are framed as testable hypotheses based on the observed cluster profiles, not guaranteed outcomes.")

    data = result["clustered"].copy()
    recs = build_recommendation_table(data)
    st.subheader("Segment-specific recommendation matrix")
    st.dataframe(recs, use_container_width=True, hide_index=True)
    st.download_button("Download recommendation matrix",
                       recs.to_csv(index=False).encode("utf-8"),
                       file_name="business_recommendations.csv", mime="text/csv")

    st.subheader("Experiment planner")
    with st.form("experiment_planner"):
        c1, c2 = st.columns(2)
        with c1:
            segment = st.selectbox("Target cluster", recs["Cluster"].tolist())
            objective = st.selectbox("Experiment objective", [
                "Increase qualified enquiries", "Improve response rate",
                "Improve customer satisfaction", "Increase referral engagement",
                "Improve property recommendation relevance"
            ])
            channel = st.selectbox("Channel", ["Email", "Website", "Sales outreach", "Paid campaign", "Referral programme"])
        with c2:
            metric = st.selectbox("Primary success metric", [
                "Qualified enquiry rate", "Click-through rate", "Conversion rate",
                "Satisfaction score", "Referral rate", "Cost per qualified enquiry"
            ])
            duration = st.selectbox("Test duration", ["2 weeks", "4 weeks", "6 weeks", "8 weeks"])
            owner = st.text_input("Owner / responsible person", placeholder="e.g., Marketing analyst")
        hypothesis = st.text_area("Hypothesis", placeholder="Example: segment-specific property information will increase qualified enquiries.")
        submitted = st.form_submit_button("Generate experiment brief")
    if submitted:
        row = recs.loc[recs["Cluster"] == segment].iloc[0]
        brief = f"""# Marketing Experiment Brief

- **Target cluster:** {segment}
- **Segment size:** {int(row['Client count'])} clients ({row['Share (%)']}%)
- **Objective:** {objective}
- **Channel:** {channel}
- **Primary metric:** {metric}
- **Duration:** {duration}
- **Owner:** {owner or 'To be assigned'}
- **Hypothesis:** {hypothesis or 'To be defined'}

## Proposed approach
1. Define eligibility rules for the selected segment.
2. Establish a baseline using comparable historical data.
3. Create a segment-specific intervention and a control group where feasible.
4. Track the primary metric and guardrail metrics consistently.
5. Compare outcomes, document limitations, and decide whether to iterate or scale.

## Candidate segment actions
{row['Suggested actions']}

## Guardrails
- Respect consent, privacy, and applicable marketing rules.
- Avoid sensitive or unsupported inferences about individual buyers.
- Do not treat cluster membership as proof of income, creditworthiness, or purchase intent.
- Confirm the experiment design and sample size with the business team before launch.
"""
        st.markdown(brief)
        st.download_button("Download experiment brief",
                           brief.encode("utf-8"),
                           file_name=f"experiment_brief_cluster_{segment}.md",
                           mime="text/markdown")

    st.subheader("Prioritized implementation roadmap")
    roadmap = pd.DataFrame([
        ["1", "Validate cluster profiles with Parcl / domain stakeholders", "High", "Business owner review"],
        ["2", "Confirm missing fields and definitions, especially transaction/value semantics", "High", "Data owner"],
        ["3", "Run a small controlled segment-based marketing experiment", "High", "Marketing analyst"],
        ["4", "Track enquiries, conversions, satisfaction and costs over time", "High", "Analytics team"],
        ["5", "Reassess cluster stability as new records arrive", "Medium", "Data science team"],
    ], columns=["Priority", "Action", "Importance", "Suggested owner"])
    st.dataframe(roadmap, use_container_width=True, hide_index=True)
    st.warning("Recommendations are proposals for validation. The current data does not establish causal impact or provide explicit income and verified investment value.")
