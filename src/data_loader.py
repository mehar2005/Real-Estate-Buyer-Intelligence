from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

@st.cache_data(show_spinner=False)
def _read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def _read_local_csv(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string)

def load_uploaded_or_default_data(clients_upload=None, properties_upload=None):
    clients = None
    properties = None
    notes = []

    clients_path = RAW_DIR / "clients.csv"
    properties_path = RAW_DIR / "properties.csv"

    try:
        if clients_upload is not None:
            clients = _read_csv_bytes(clients_upload.getvalue())
            notes.append("clients.csv loaded from upload")
        elif clients_path.exists():
            clients = _read_local_csv(str(clients_path))
            notes.append("clients.csv loaded from data/raw/")
    except Exception as exc:
        st.error(f"Could not read clients dataset: {exc}")

    try:
        if properties_upload is not None:
            properties = _read_csv_bytes(properties_upload.getvalue())
            notes.append("properties.csv loaded from upload")
        elif properties_path.exists():
            properties = _read_local_csv(str(properties_path))
            notes.append("properties.csv loaded from data/raw/")
    except Exception as exc:
        st.error(f"Could not read properties dataset: {exc}")

    return clients, properties, " · ".join(notes) if notes else "No dataset loaded"
