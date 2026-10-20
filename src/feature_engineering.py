import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

ID_COLUMNS = {"client_id", "first_name", "last_name", "date_of_birth", "client_ref", "listing_id"}

def build_client_features(clients, properties):
    clients = clients.copy()
    properties = properties.copy()
    notes = []

    # Aggregate property-level activity to one row per client.
    property_aggregates = pd.DataFrame(columns=["client_ref"])
    if "client_ref" in properties.columns:
        props = properties.copy()
        props["client_ref"] = props["client_ref"].astype("string")
        agg_spec = {}
        if "listing_id" in props:
            agg_spec["listing_id"] = "count"
        if "sale_price" in props:
            agg_spec["sale_price"] = ["mean", "median", "sum"]
        if "floor_area_sqft" in props:
            agg_spec["floor_area_sqft"] = ["mean", "sum"]
        if "listing_status" in props:
            # Create a stable sold indicator rather than assuming every row is a sale.
            props["is_sold"] = props["listing_status"].astype("string").str.lower().eq("sold").astype(int)
            agg_spec["is_sold"] = "sum"
        if agg_spec:
            property_aggregates = props.groupby("client_ref", dropna=False).agg(agg_spec)
            property_aggregates.columns = [
                "_".join(str(part) for part in col if part)
                if isinstance(col, tuple) else str(col)
                for col in property_aggregates.columns
            ]
            property_aggregates = property_aggregates.reset_index()
            property_aggregates = property_aggregates.rename(columns={
                "listing_id_count": "property_record_count",
                "sale_price_mean": "mean_property_price",
                "sale_price_median": "median_property_price",
                "sale_price_sum": "total_recorded_property_price",
                "floor_area_sqft_mean": "mean_floor_area_sqft",
                "floor_area_sqft_sum": "total_floor_area_sqft",
                "is_sold_sum": "sold_listing_count",
            })
            property_aggregates["client_ref"] = property_aggregates["client_ref"].astype("string")
            notes.append("Property activity aggregated to client level using client_ref.")
    else:
        notes.append("No client_ref field: property activity could not be joined.")

    if "client_id" in clients.columns:
        clients["client_id"] = clients["client_id"].astype("string")
        features = clients.merge(
            property_aggregates, left_on="client_id", right_on="client_ref",
            how="left", validate="one_to_one"
        )
    else:
        features = clients.copy()
        notes.append("No client_id field: client-level join unavailable.")

    # Explicit property counts should be zero when a client has no matched property row.
    for col in ["property_record_count", "sold_listing_count", "total_recorded_property_price",
                "total_floor_area_sqft"]:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    # Derived features: age is calculated during cleaning; transaction aggregates above.
    if "satisfaction_score" in features.columns:
        features["satisfaction_score"] = pd.to_numeric(features["satisfaction_score"], errors="coerce")
    if "age" in features.columns:
        features.loc[~features["age"].between(18, 110), "age"] = np.nan

    # Remove direct identifiers and names from the modeling table; keep client_id only for output mapping.
    return features, notes

def _feature_columns(df):
    excluded = ID_COLUMNS | {"client_id"}
    numeric = [
        c for c in df.select_dtypes(include=["number"]).columns
        if c not in excluded and df[c].notna().any()
    ]
    categorical = [
        c for c in df.select_dtypes(include=["object", "string", "category", "bool"]).columns
        if c not in excluded and df[c].nunique(dropna=True) > 1
    ]
    return numeric, categorical

def build_model_matrix(features):
    numeric, categorical = _feature_columns(features)
    if not numeric and not categorical:
        return None, None, {"numeric": [], "categorical": []}

    transformers = []
    if numeric:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipe, numeric))
    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipe, categorical))

    preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    try:
        matrix = preprocessor.fit_transform(features)
    except TypeError:
        # Compatibility fallback for older scikit-learn versions.
        if categorical:
            categorical_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ])
            transformers = []
            if numeric:
                transformers.append(("numeric", Pipeline([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                ]), numeric))
            transformers.append(("categorical", categorical_pipe, categorical))
            preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
            matrix = preprocessor.fit_transform(features)
        else:
            raise
    metadata = {"numeric": numeric, "categorical": categorical}
    return np.asarray(matrix, dtype=float), preprocessor, metadata

def render_feature_engineering(features, notes):
    st.header("Section 4 — Feature Engineering")
    st.write("The modeling table is built at the client level. Property records are aggregated by client reference so that customers with multiple property records are represented by summary features.")
    if notes:
        for note in notes:
            st.caption(f"• {note}")

    numeric, categorical = _feature_columns(features)
    c1, c2, c3 = st.columns(3)
    c1.metric("Client rows", f"{len(features):,}")
    c2.metric("Numeric features", len(numeric))
    c3.metric("Categorical features", len(categorical))

    st.subheader("Engineered feature definitions")
    descriptions = {
        "age": "Age derived from date_of_birth using the current date; implausible ages are set missing.",
        "property_record_count": "Number of property records linked to the client.",
        "sold_listing_count": "Number of linked property records whose listing_status is Sold.",
        "mean_property_price": "Mean numeric sale_price among linked property records.",
        "median_property_price": "Median numeric sale_price among linked property records.",
        "total_recorded_property_price": "Sum of recorded property prices; this is not guaranteed to be a client's investment total.",
        "mean_floor_area_sqft": "Mean floor area among linked property records.",
        "total_floor_area_sqft": "Sum of linked property floor areas.",
        "satisfaction_score": "Customer satisfaction score from the supplied client data.",
    }
    rows = []
    for col in features.columns:
        if col in {"client_id", "client_ref", "first_name", "last_name", "date_of_birth"}:
            role = "Identifier / excluded from model"
        elif col in numeric:
            role = "Numeric model feature"
        elif col in categorical:
            role = "Categorical model feature"
        else:
            role = "Context / excluded"
        rows.append({"Feature": col, "Role": role, "Description": descriptions.get(col, "Original dataset field.")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Preview client-level modeling data"):
        st.dataframe(features.head(100), use_container_width=True, hide_index=True)
        st.download_button(
            "Download client-level features CSV",
            features.to_csv(index=False).encode("utf-8"),
            file_name="client_level_features.csv",
            mime="text/csv",
        )

    st.subheader("Encoding and scaling strategy")
    st.markdown("""
    - **Numeric features:** median imputation followed by `StandardScaler`.
    - **Categorical features:** most-frequent imputation followed by `OneHotEncoder(handle_unknown='ignore')`.
    - **Identifiers and names:** excluded from clustering to prevent arbitrary IDs or personal names from driving similarity.
    - **Missing values:** imputed within the preprocessing pipeline so the model matrix can be generated consistently.
    """)
    st.write("Numeric fields:", ", ".join(numeric) if numeric else "None")
    st.write("Categorical fields:", ", ".join(categorical) if categorical else "None")
