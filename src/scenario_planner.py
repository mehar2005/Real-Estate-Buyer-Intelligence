import pandas as pd
import plotly.express as px
import streamlit as st


def render_scenario_planner(result):
    st.header("Section 18 — Segment Campaign Scenario Planner")
    st.write("Estimate the potential economics of outreach campaigns by buyer segment. All rates and costs are user-entered assumptions, not predictions from the clustering model.")
    clustered = result["clustered"].copy()
    counts = clustered.groupby("cluster").size().rename("available_clients").reset_index()
    counts["cluster"] = counts["cluster"].astype(str)
    selected = st.selectbox("Target buyer segment", counts["cluster"].tolist(), key="scenario_cluster")
    available = int(counts.loc[counts["cluster"] == selected, "available_clients"].iloc[0])

    a, b, c = st.columns(3)
    with a:
        reach_pct = st.slider("Share of segment contacted (%)", 1, 100, 40, key="scenario_reach")
        response_pct = st.slider("Expected response rate (%)", 0, 100, 12, key="scenario_response")
    with b:
        conversion_pct = st.slider("Response-to-conversion rate (%)", 0, 100, 20, key="scenario_conversion")
        cost_per_contact = st.number_input("Cost per contacted buyer", min_value=0.0, value=150.0, step=25.0, key="scenario_cost")
    with c:
        value_per_conversion = st.number_input("Estimated value per conversion", min_value=0.0, value=25000.0, step=1000.0, key="scenario_value")
        variable_cost_pct = st.slider("Variable cost of conversion value (%)", 0, 100, 35, key="scenario_variable_cost")

    contacted = round(available * reach_pct / 100)
    responses = round(contacted * response_pct / 100)
    conversions = round(responses * conversion_pct / 100)
    campaign_cost = contacted * cost_per_contact
    gross_value = conversions * value_per_conversion
    variable_cost = gross_value * variable_cost_pct / 100
    estimated_net = gross_value - variable_cost - campaign_cost
    roi = (estimated_net / campaign_cost * 100) if campaign_cost > 0 else None

    st.subheader("Scenario results")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Buyers contacted", f"{contacted:,}")
    k2.metric("Estimated conversions", f"{conversions:,}")
    k3.metric("Estimated net value", f"{estimated_net:,.2f}")
    k4.metric("Estimated ROI", f"{roi:.1f}%" if roi is not None else "N/A")

    breakdown = pd.DataFrame({"Measure": ["Contacted buyers", "Responses", "Conversions"],
                              "Count": [contacted, responses, conversions]})
    st.plotly_chart(px.bar(breakdown, x="Measure", y="Count", text="Count", title="Illustrative campaign funnel"), use_container_width=True)
    st.dataframe(pd.DataFrame({"Metric": ["Segment size", "Contact rate", "Response rate", "Conversion rate", "Campaign cost", "Gross conversion value", "Variable costs", "Estimated net value", "Estimated ROI"],
                               "Value": [available, f"{reach_pct}%", f"{response_pct}%", f"{conversion_pct}%", round(campaign_cost, 2), round(gross_value, 2), round(variable_cost, 2), round(estimated_net, 2), f"{roi:.2f}%" if roi is not None else "N/A"]}), use_container_width=True, hide_index=True)
    export = pd.DataFrame([{"cluster": selected, "available_clients": available, "reach_pct": reach_pct,
        "response_pct_assumption": response_pct, "conversion_pct_assumption": conversion_pct,
        "contacted_buyers": contacted, "estimated_responses": responses, "estimated_conversions": conversions,
        "cost_per_contact": cost_per_contact, "value_per_conversion": value_per_conversion,
        "variable_cost_pct": variable_cost_pct, "campaign_cost": campaign_cost,
        "gross_value": gross_value, "variable_cost": variable_cost, "estimated_net_value": estimated_net,
        "estimated_roi_pct": roi}])
    st.download_button("Download scenario assumptions and results", export.to_csv(index=False).encode("utf-8"),
                       file_name=f"campaign_scenario_cluster_{selected}.csv", mime="text/csv")
    st.warning("Use this only for planning sensitivity. Validate response, conversion, cost, and value assumptions with actual campaign results before making investment or budget decisions.")
