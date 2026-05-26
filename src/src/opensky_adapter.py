import pandas as pd
import numpy as np


def convert_opensky_arrivals_to_dashboard_format(raw_df):
    """
    Convert OpenSky airport-arrival records into the dashboard's standard
    arrival-data schema.

    Expected OpenSky-style columns may include:
    - icao24
    - callsign
    - firstSeen
    - lastSeen
    - estDepartureAirport
    - estArrivalAirport
    - estArrivalAirportHorizDistance
    - estArrivalAirportVertDistance

    Notes
    -----
    OpenSky airport-arrival records do not always contain full speed/altitude
    profiles. Therefore, this converter creates transparent approximations
    needed by the green-arrival optimization prototype.
    """

    df = raw_df.copy()

    # Standard aircraft identifier
    if "callsign" in df.columns:
        aircraft_id = df["callsign"].fillna("").astype(str).str.strip()
        aircraft_id = aircraft_id.replace("", np.nan)
    else:
        aircraft_id = pd.Series([np.nan] * len(df))

    if "icao24" in df.columns:
        fallback_id = df["icao24"].fillna("UNKNOWN").astype(str)
    else:
        fallback_id = pd.Series([f"AC_{i+1}" for i in range(len(df))])

    aircraft_id = aircraft_id.fillna(fallback_id)

    # Time conversion
    # OpenSky firstSeen and lastSeen are Unix timestamps in seconds.
    if "firstSeen" in df.columns:
        timestamp = pd.to_datetime(df["firstSeen"], unit="s", errors="coerce")
    elif "timestamp" in df.columns:
        timestamp = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        timestamp = pd.date_range(
            start="2026-01-01 08:00:00",
            periods=len(df),
            freq="min"
        )

    if "lastSeen" in df.columns:
        estimated_arrival_time = pd.to_datetime(df["lastSeen"], unit="s", errors="coerce")
    elif "estimated_arrival_time" in df.columns:
        estimated_arrival_time = pd.to_datetime(df["estimated_arrival_time"], errors="coerce")
    else:
        estimated_arrival_time = timestamp + pd.Timedelta(minutes=20)

    # Distance to airport
    # OpenSky estArrivalAirportHorizDistance is in meters.
    if "estArrivalAirportHorizDistance" in df.columns:
        distance_to_airport_km = pd.to_numeric(
            df["estArrivalAirportHorizDistance"],
            errors="coerce"
        ) / 1000.0
    elif "distance_to_airport_km" in df.columns:
        distance_to_airport_km = pd.to_numeric(
            df["distance_to_airport_km"],
            errors="coerce"
        )
    else:
        distance_to_airport_km = pd.Series(
            np.linspace(160, 40, len(df))
        )

    # Some arrival records can have very small final horizontal distance.
    # For sequencing experiments, create a conservative distance proxy if needed.
    distance_to_airport_km = distance_to_airport_km.fillna(distance_to_airport_km.median())
    distance_to_airport_km = distance_to_airport_km.clip(lower=1, upper=300)

    # Altitude approximation
    # OpenSky airport-arrival endpoint usually does not provide full altitude history.
    # estArrivalAirportVertDistance is in meters if present.
    if "estArrivalAirportVertDistance" in df.columns:
        altitude_ft = pd.to_numeric(
            df["estArrivalAirportVertDistance"],
            errors="coerce"
        ) * 3.28084
    elif "altitude_ft" in df.columns:
        altitude_ft = pd.to_numeric(df["altitude_ft"], errors="coerce")
    else:
        altitude_ft = 2500 + distance_to_airport_km * 180

    altitude_ft = pd.Series(altitude_ft).fillna(pd.Series(altitude_ft).median())
    altitude_ft = altitude_ft.clip(lower=0, upper=45000)

    # Speed approximation
    # The airport-arrival endpoint does not reliably provide speed.
    # Use distance/time approximation if possible.
    duration_min = (
        (estimated_arrival_time - timestamp).dt.total_seconds() / 60.0
    )

    duration_min = duration_min.replace(0, np.nan).fillna(20).clip(lower=5, upper=180)

    speed_kmh = distance_to_airport_km / (duration_min / 60.0)
    speed_kt = speed_kmh / 1.852
    speed_kt = speed_kt.clip(lower=120, upper=520)

    # Route angle is not available from basic arrival record.
    # Use deterministic pseudo-angle for visualization.
    route_angle_deg = (
        pd.Series(range(len(df))) * 37
    ) % 360

    # Runway is not available in OpenSky arrival records.
    runway = "UNKNOWN"

    converted = pd.DataFrame(
        {
            "aircraft_id": aircraft_id,
            "timestamp": timestamp,
            "distance_to_airport_km": distance_to_airport_km.round(2),
            "altitude_ft": altitude_ft.round(0).astype(int),
            "speed_kt": speed_kt.round(1),
            "estimated_arrival_time": estimated_arrival_time,
            "route_angle_deg": route_angle_deg,
            "runway": runway,
        }
    )

    converted = converted.dropna(subset=["timestamp", "estimated_arrival_time"])
    converted = converted.sort_values("estimated_arrival_time").reset_index(drop=True)

    return converted


def is_opensky_arrival_format(df):
    """
    Heuristic check for OpenSky airport-arrival format.
    """

    opensky_columns = {
        "icao24",
        "firstSeen",
        "lastSeen",
        "estDepartureAirport",
        "estArrivalAirport",
        "callsign",
    }

    return len(opensky_columns.intersection(set(df.columns))) >= 3
