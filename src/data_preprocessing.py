import pandas as pd
import numpy as np


STANDARD_COLUMNS = [
    "aircraft_id",
    "timestamp",
    "distance_to_airport_km",
    "altitude_ft",
    "speed_kt",
    "estimated_arrival_time",
    "route_angle_deg",
    "runway",
]


def load_uploaded_data(uploaded_file):
    """
    Load uploaded aviation data from CSV, Excel, or JSON.
    """

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    if file_name.endswith(".json"):
        return pd.read_json(uploaded_file)

    raise ValueError("Unsupported file format. Please upload CSV, Excel, or JSON.")


def suggest_column_mapping(df):
    """
    Suggest likely mappings from user dataset columns to standard dashboard columns.
    """

    columns = list(df.columns)
    lower_map = {col.lower(): col for col in columns}

    suggestions = {}

    aliases = {
        "aircraft_id": ["aircraft_id", "aircraft", "callsign", "flight_id", "flight", "icao24"],
        "timestamp": ["timestamp", "time", "datetime", "date_time", "observed_time"],
        "distance_to_airport_km": ["distance_to_airport_km", "distance", "distance_km", "dist_km"],
        "altitude_ft": ["altitude_ft", "altitude", "baro_altitude", "geoaltitude", "alt"],
        "speed_kt": ["speed_kt", "speed", "velocity", "groundspeed", "ground_speed"],
        "estimated_arrival_time": ["estimated_arrival_time", "eta", "arrival_time", "estimated_time"],
        "route_angle_deg": ["route_angle_deg", "heading", "track", "bearing", "angle"],
        "runway": ["runway", "rwy", "arrival_runway"],
    }

    for standard_col, possible_names in aliases.items():
        suggestions[standard_col] = None

        for name in possible_names:
            if name.lower() in lower_map:
                suggestions[standard_col] = lower_map[name.lower()]
                break

    return suggestions


def apply_column_mapping(df, column_mapping):
    """
    Rename user-selected columns to the standard names used by the dashboard.
    """

    mapped_df = pd.DataFrame()

    for standard_col, actual_col in column_mapping.items():
        if actual_col is not None and actual_col != "Not available":
            mapped_df[standard_col] = df[actual_col]

    return mapped_df


def clean_aviation_data(df):
    """
    Clean aviation arrival data.

    Handles missing values, wrong data types, duplicate rows, and simple outliers.
    Returns cleaned dataframe and a cleaning report.
    """

    cleaned = df.copy()
    report = {}

    report["initial_rows"] = len(cleaned)
    report["initial_columns"] = len(cleaned.columns)
    report["initial_missing_values"] = int(cleaned.isna().sum().sum())
    report["duplicate_rows_removed"] = int(cleaned.duplicated().sum())

    cleaned = cleaned.drop_duplicates()

    # Ensure required columns exist where possible
    if "aircraft_id" in cleaned.columns:
        cleaned["aircraft_id"] = cleaned["aircraft_id"].fillna("UNKNOWN")
    else:
        cleaned["aircraft_id"] = [f"AC_{i+1}" for i in range(len(cleaned))]

    # Time columns
    if "timestamp" in cleaned.columns:
        cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")

    if "estimated_arrival_time" in cleaned.columns:
        cleaned["estimated_arrival_time"] = pd.to_datetime(
            cleaned["estimated_arrival_time"],
            errors="coerce"
        )

    # Drop rows without timestamp
    if "timestamp" in cleaned.columns:
        missing_timestamp_rows = int(cleaned["timestamp"].isna().sum())
        cleaned = cleaned.dropna(subset=["timestamp"])
    else:
        missing_timestamp_rows = 0

    report["rows_removed_missing_timestamp"] = missing_timestamp_rows

    # Numeric columns
    numeric_columns = [
        "distance_to_airport_km",
        "altitude_ft",
        "speed_kt",
        "route_angle_deg",
    ]

    for col in numeric_columns:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Outlier clipping rules
    outlier_report = {}

    bounds = {
        "distance_to_airport_km": (0, 300),
        "altitude_ft": (0, 45000),
        "speed_kt": (50, 600),
        "route_angle_deg": (0, 360),
    }

    for col, (lower, upper) in bounds.items():
        if col in cleaned.columns:
            before_outliers = int(((cleaned[col] < lower) | (cleaned[col] > upper)).sum())
            cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)
            outlier_report[col] = before_outliers

    report["outliers_clipped"] = outlier_report

    # Missing numeric values: median imputation
    imputed_values = {}

    for col in numeric_columns:
        if col in cleaned.columns:
            missing_before = int(cleaned[col].isna().sum())

            if missing_before > 0:
                median_value = cleaned[col].median()

                if pd.isna(median_value):
                    median_value = 0

                cleaned[col] = cleaned[col].fillna(median_value)
                imputed_values[col] = {
                    "missing_values_imputed": missing_before,
                    "imputation_value": float(median_value)
                }

    report["numeric_imputation"] = imputed_values

    # Estimated arrival time fallback
    if "estimated_arrival_time" in cleaned.columns and "timestamp" in cleaned.columns:
        missing_eta = cleaned["estimated_arrival_time"].isna()

        if missing_eta.any():
            cleaned.loc[missing_eta, "estimated_arrival_time"] = (
                cleaned.loc[missing_eta, "timestamp"] + pd.Timedelta(minutes=20)
            )

        report["estimated_arrival_time_imputed"] = int(missing_eta.sum())

    # Runway fallback
    if "runway" in cleaned.columns:
        runway_missing = int(cleaned["runway"].isna().sum())
        cleaned["runway"] = cleaned["runway"].fillna("UNKNOWN")
        report["runway_missing_filled"] = runway_missing
    else:
        cleaned["runway"] = "UNKNOWN"
        report["runway_missing_filled"] = len(cleaned)

    report["final_rows"] = len(cleaned)
    report["final_missing_values"] = int(cleaned.isna().sum().sum())

    return cleaned, report


def validate_required_columns(df):
    """
    Check which standard columns are present or missing.
    """

    present = [col for col in STANDARD_COLUMNS if col in df.columns]
    missing = [col for col in STANDARD_COLUMNS if col not in df.columns]

    return present, missing
