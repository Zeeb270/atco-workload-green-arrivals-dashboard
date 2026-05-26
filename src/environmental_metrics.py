import numpy as np
import pandas as pd


def infer_aircraft_weight_factor(aircraft_id):
    """
    Estimate a simple aircraft weight/fuel factor from the aircraft identifier.

    This is a proxy. In real work, this should be replaced with aircraft-type data,
    BADA/OpenAP-style aircraft performance data, or airline fleet information.
    """

    if pd.isna(aircraft_id):
        return 1.0

    aircraft_id = str(aircraft_id).upper()

    # Very rough airline/operator-based proxy.
    # The aim is only to create heterogeneous environmental sensitivity.
    heavy_prefixes = ["DLH", "BAW", "AFR", "KLM", "SWR", "QTR", "UAE", "THY"]
    medium_prefixes = ["SAS", "FIN", "LOT", "AUA", "IBE", "TAP"]
    low_cost_prefixes = ["RYR", "EZY", "NAX", "WZZ"]

    if any(aircraft_id.startswith(prefix) for prefix in heavy_prefixes):
        return 1.20

    if any(aircraft_id.startswith(prefix) for prefix in medium_prefixes):
        return 1.00

    if any(aircraft_id.startswith(prefix) for prefix in low_cost_prefixes):
        return 0.90

    return 1.00


def compute_descent_inefficiency(row):
    """
    Estimate descent inefficiency from distance and altitude.

    A simple idealized descent profile is assumed:
    altitude should decrease approximately linearly with distance to airport.

    This is not a certified flight-performance model. It is a transparent proxy
    for research prototyping.
    """

    distance = row.get("distance_to_airport_km", 0)
    altitude = row.get("altitude_ft", 0)

    # Idealized descent slope:
    # about 300 ft per nautical mile.
    # 1 km = 0.5399568 nautical miles.
    ideal_altitude = distance * 0.5399568 * 300

    # Add lower bound because aircraft close to airport may still be above field elevation.
    ideal_altitude = max(ideal_altitude, 2500)

    deviation = abs(altitude - ideal_altitude)

    # Scale to a manageable score.
    return deviation / 10000.0


def compute_speed_inefficiency(row):
    """
    Estimate speed inefficiency.

    Aircraft far from the airport can be faster; aircraft close to the airport
    should generally be slower. This creates a transparent speed-profile proxy.
    """

    distance = row.get("distance_to_airport_km", 0)
    speed = row.get("speed_kt", 0)

    if distance > 120:
        reference_speed = 450
    elif distance > 70:
        reference_speed = 360
    elif distance > 30:
        reference_speed = 280
    else:
        reference_speed = 220

    return abs(speed - reference_speed) / 100.0


def add_environmental_metrics(scheduled_df):
    """
    Add improved environmental metrics to an aircraft-level schedule.

    Parameters
    ----------
    scheduled_df : pandas.DataFrame
        Aircraft-level schedule containing delay, holding, altitude, speed,
        distance, and aircraft ID.

    Returns
    -------
    pandas.DataFrame
        Schedule with environmental metrics.
    """

    df = scheduled_df.copy()

    if "holding_proxy_min" not in df.columns:
        df["holding_proxy_min"] = df.get("delay_min", 0).clip(lower=0)

    if "extra_distance_proxy_km" not in df.columns:
        df["extra_distance_proxy_km"] = df["holding_proxy_min"] * (
            df["speed_kt"] * 1.852 / 60.0
        )

    if "level_flight_proxy_min" not in df.columns:
        low_altitude_factor = np.where(
            df["altitude_ft"] < 12000,
            1.5,
            np.where(df["altitude_ft"] < 24000, 1.2, 1.0)
        )
        df["level_flight_proxy_min"] = df["holding_proxy_min"] * low_altitude_factor

    df["descent_inefficiency_score"] = df.apply(compute_descent_inefficiency, axis=1)
    df["speed_inefficiency_score"] = df.apply(compute_speed_inefficiency, axis=1)

    df["aircraft_weight_factor"] = df["aircraft_id"].apply(infer_aircraft_weight_factor)

    # Fuel-burn proxy:
    # weights are chosen to reflect that holding and low-altitude level flight
    # are environmentally costly, while descent/speed inefficiency contribute
    # additional penalties.
    df["fuel_burn_proxy"] = df["aircraft_weight_factor"] * (
        4.0 * df["holding_proxy_min"]
        + 0.25 * df["extra_distance_proxy_km"]
        + 3.0 * df["level_flight_proxy_min"]
        + 6.0 * df["descent_inefficiency_score"]
        + 2.0 * df["speed_inefficiency_score"]
    )

    # CO2 proxy:
    # Jet fuel combustion produces about 3.16 kg CO2 per kg fuel.
    # Since fuel_burn_proxy is not actual kg fuel, this remains a scaled proxy.
    df["co2_proxy"] = 3.16 * df["fuel_burn_proxy"]

    df["environmental_cost"] = (
        0.60 * df["fuel_burn_proxy"]
        + 0.40 * df["co2_proxy"]
    )

    return df


def summarize_environmental_metrics(scheduled_df):
    """
    Summarize environmental metrics at strategy level.
    """

    df = scheduled_df.copy()

    return {
        "fuel_burn_proxy": round(df["fuel_burn_proxy"].sum(), 2),
        "co2_proxy": round(df["co2_proxy"].sum(), 2),
        "environmental_cost": round(df["environmental_cost"].sum(), 2),
        "avg_descent_inefficiency": round(df["descent_inefficiency_score"].mean(), 3),
        "avg_speed_inefficiency": round(df["speed_inefficiency_score"].mean(), 3),
    }
