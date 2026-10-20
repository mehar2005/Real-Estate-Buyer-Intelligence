import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def _safe_count_plot(df, column, title, top_n=15):
    if column not in df.columns:
        st.info(f"`{column}` is not available in this dataset.")
        return
    counts = df[column].fillna("Missing").astype(str).value_counts().head(top_n).reset_index()
    counts.columns = [column, "Count"]
    fig = px.bar(counts, x=column, y="Count", title=title, text="Count")
    fig.update_layout(xaxis_title=column.replace("_", " ").title(), yaxis_title="Records")
    st.plotly_chart(fig, use_container_width=True)

def render_eda(clients, properties, client_features):
    st.header("Exploratory Data Analysis")
    st.write("Explore the distributions, missingness, and relationships in the supplied client and property datasets before fitting a clustering model.")

    t1, t2, t3 = st.tabs(["Client profiles", "Property activity", "Data quality"])
    with t1:
        a, b, c, d = st.columns(4)
        a.metric("Clients", f"{len(clients):,}")
        b.metric("Countries", f"{clients['country'].nunique(dropna=True):,}" if "country" in clients else "—")
        c.metric("Regions", f"{clients['region'].nunique(dropna=True):,}" if "region" in clients else "—")
        d.metric("Avg. satisfaction", f"{clients['satisfaction_score'].mean():.2f}" if "satisfaction_score" in clients else "—")
        left, right = st.columns(2)
        with left:
            _safe_count_plot(clients, "client_type", "Client type distribution")
            _safe_count_plot(clients, "acquisition_purpose", "Acquisition purpose")
            _safe_count_plot(clients, "loan_applied", "Loan application pattern")
        with right:
            _safe_count_plot(clients, "country", "Clients by country", 12)
            _safe_count_plot(clients, "referral_channel", "Referral channel")
            if "satisfaction_score" in clients:
                sat = clients["satisfaction_score"].value_counts(dropna=False).sort_index().reset_index()
                sat.columns = ["Satisfaction score", "Count"]
                fig = px.bar(sat, x="Satisfaction score", y="Count", title="Satisfaction score distribution")
                st.plotly_chart(fig, use_container_width=True)
        if "age" in clients:
            fig = px.histogram(clients, x="age", nbins=25, title="Client age distribution")
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        a, b, c = st.columns(3)
        a.metric("Property records", f"{len(properties):,}")
        b.metric("Unit categories", f"{properties['unit_category'].nunique(dropna=True):,}" if "unit_category" in properties else "—")
        c.metric("Median floor area", f"{properties['floor_area_sqft'].median():,.0f} sq ft" if "floor_area_sqft" in properties else "—")
        left, right = st.columns(2)
        with left:
            _safe_count_plot(properties, "listing_status", "Listing status")
            _safe_count_plot(properties, "unit_category", "Unit category")
        with right:
            if "sale_price" in properties:
                valid = properties.dropna(subset=["sale_price"])
                if not valid.empty:
                    fig = px.histogram(valid, x="sale_price", nbins=35, title="Sale price distribution")
                    st.plotly_chart(fig, use_container_width=True)
            if {"floor_area_sqft", "sale_price"}.issubset(properties.columns):
                valid = properties.dropna(subset=["floor_area_sqft", "sale_price"])
                if not valid.empty:
                    fig = px.scatter(valid, x="floor_area_sqft", y="sale_price",
                                     color="unit_category" if "unit_category" in valid else None,
                                     opacity=0.55, title="Floor area vs. sale price")
                    st.plotly_chart(fig, use_container_width=True)
        if "transaction_date" in properties:
            dated = properties.dropna(subset=["transaction_date"]).copy()
            if not dated.empty:
                dated["month"] = dated["transaction_date"].dt.to_period("M").astype(str)
                monthly = dated.groupby("month").size().reset_index(name="Records")
                fig = px.line(monthly, x="month", y="Records", markers=True, title="Property records over time")
                st.plotly_chart(fig, use_container_width=True)

    with t3:
        st.subheader("Missing values")
        missing = []
        for name, df in [("Clients", clients), ("Properties", properties)]:
            for col in df.columns:
                missing.append({"Dataset": name, "Column": col,
                                "Missing": int(df[col].isna().sum()),
                                "Missing %": round(float(df[col].isna().mean() * 100), 2)})
        missing_df = pd.DataFrame(missing).sort_values(["Dataset", "Missing %"], ascending=[True, False])
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
        fig = px.bar(missing_df, x="Column", y="Missing %", color="Dataset",
                     barmode="group", title="Missing values by column (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Charts describe the supplied data; they do not imply causation. Small categories and missing values should be reviewed before making business decisions.")
