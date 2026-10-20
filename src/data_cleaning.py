import pandas as pd
import numpy as np

CLIENT_REQUIRED = [
    "client_id", "client_type", "date_of_birth", "gender", "country", "region",
    "acquisition_purpose", "satisfaction_score", "loan_applied", "referral_channel"
]
PROPERTY_REQUIRED = [
    "listing_id", "transaction_date", "unit_category", "floor_area_sqft",
    "sale_price", "listing_status", "client_ref"
]

def _normalise_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

def clean_clients(df: pd.DataFrame):
    cleaned = df.copy()
    before_rows = len(cleaned)
    before_duplicates = int(cleaned.duplicated().sum())

    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = _normalise_text(cleaned[col])

    # Standardise common category labels while preserving the original meaning.
    category_cols = [
        "client_type", "gender", "country", "region", "acquisition_purpose",
        "loan_applied", "referral_channel"
    ]
    for col in category_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype("string").str.strip()
            cleaned[col] = cleaned[col].str.replace(r"\s+", " ", regex=True)
            cleaned[col] = cleaned[col].str.title()

    if "client_id" in cleaned.columns:
        cleaned["client_id"] = cleaned["client_id"].astype("string").str.strip()
        cleaned = cleaned.drop_duplicates(subset=["client_id"], keep="first")
    else:
        cleaned = cleaned.drop_duplicates()

    if "date_of_birth" in cleaned.columns:
        # Day-first parsing supports the formats present in the supplied dataset.
        raw_dob = cleaned["date_of_birth"].astype("string").str.strip()
        # The supplied file mixes day-first hyphen dates and month-first slash dates.
        parsed_dob = pd.Series(pd.NaT, index=cleaned.index, dtype="datetime64[ns]")
        hyphen_mask = raw_dob.str.contains("-", na=False)
        slash_mask = raw_dob.str.contains("/", na=False)
        parsed_dob.loc[hyphen_mask] = pd.to_datetime(
            raw_dob.loc[hyphen_mask], errors="coerce", dayfirst=True, format="mixed"
        )
        parsed_dob.loc[slash_mask] = pd.to_datetime(
            raw_dob.loc[slash_mask], errors="coerce", dayfirst=False, format="mixed"
        )
        other_mask = ~(hyphen_mask | slash_mask)
        parsed_dob.loc[other_mask] = pd.to_datetime(
            raw_dob.loc[other_mask], errors="coerce", format="mixed"
        )
        cleaned["date_of_birth"] = parsed_dob
        reference_date = pd.Timestamp.today().normalize()
        age = (reference_date - cleaned["date_of_birth"]).dt.days / 365.2425
        cleaned["age"] = age.where(age.between(18, 110)).round(1)

    if "satisfaction_score" in cleaned.columns:
        cleaned["satisfaction_score"] = pd.to_numeric(
            cleaned["satisfaction_score"], errors="coerce"
        )
        cleaned.loc[
            ~cleaned["satisfaction_score"].between(1, 5),
            "satisfaction_score"
        ] = np.nan

    if "loan_applied" in cleaned.columns:
        cleaned["loan_applied"] = cleaned["loan_applied"].replace({
            "Y": "Yes", "N": "No", "True": "Yes", "False": "No"
        })

    report = _quality_report(df, cleaned, before_rows, before_duplicates, CLIENT_REQUIRED)
    return cleaned, report

def clean_properties(df: pd.DataFrame):
    cleaned = df.copy()
    before_rows = len(cleaned)
    before_duplicates = int(cleaned.duplicated().sum())

    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = _normalise_text(cleaned[col])

    for col in ["unit_category", "listing_status"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype("string").str.strip().str.replace(
                r"\s+", " ", regex=True
            ).str.title()

    if "listing_id" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["listing_id"], keep="first")
    else:
        cleaned = cleaned.drop_duplicates()

    if "transaction_date" in cleaned.columns:
        cleaned["transaction_date"] = pd.to_datetime(
            cleaned["transaction_date"], errors="coerce", dayfirst=True
        )

    if "floor_area_sqft" in cleaned.columns:
        cleaned["floor_area_sqft"] = pd.to_numeric(cleaned["floor_area_sqft"], errors="coerce")
        cleaned.loc[cleaned["floor_area_sqft"] <= 0, "floor_area_sqft"] = np.nan

    if "sale_price" in cleaned.columns:
        price = cleaned["sale_price"].astype("string").str.replace(r"[$,\s]", "", regex=True)
        cleaned["sale_price"] = pd.to_numeric(price, errors="coerce")
        cleaned.loc[cleaned["sale_price"] <= 0, "sale_price"] = np.nan

    if "client_ref" in cleaned.columns:
        cleaned["client_ref"] = cleaned["client_ref"].astype("string").str.strip()

    report = _quality_report(df, cleaned, before_rows, before_duplicates, PROPERTY_REQUIRED)
    return cleaned, report

def _quality_report(original, cleaned, before_rows, before_duplicates, required_columns):
    missing_required = [c for c in required_columns if c not in original.columns]
    rows = [
        ("Rows before cleaning", before_rows),
        ("Rows after cleaning", len(cleaned)),
        ("Rows removed", before_rows - len(cleaned)),
        ("Duplicate rows in original", before_duplicates),
        ("Missing cells after cleaning", int(cleaned.isna().sum().sum())),
        ("Columns with missing values", int((cleaned.isna().sum() > 0).sum())),
        ("Missing required columns", ", ".join(missing_required) if missing_required else "None"),
    ]
    for col in cleaned.columns:
        rows.append((f"Missing values — {col}", int(cleaned[col].isna().sum())))
    return pd.DataFrame(rows, columns=["Quality check", "Result"])

def build_customer_property_dataset(clients: pd.DataFrame, properties: pd.DataFrame):
    if "client_id" not in clients.columns or "client_ref" not in properties.columns:
        return None
    # Keep all client records and attach matching property activity.
    merged = properties.merge(
        clients, left_on="client_ref", right_on="client_id", how="left",
        suffixes=("_property", "_client"), indicator=True
    )
    return merged
