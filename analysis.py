
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Palo Alto Networks.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

SATISFACTION_COLUMNS = [
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
]

REQUIRED_COLUMNS = SATISFACTION_COLUMNS + [
    "WorkLifeBalance",
    "OverTime",
    "Department",
    "JobRole",
    "JobLevel",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "BusinessTravel",
    "DistanceFromHome",
    "Attrition",
]


def load_data():
    """Load and validate the employee dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place Palo Alto Networks.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Your CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    # Normalize common text fields.
    for col in ["OverTime", "Department", "JobRole", "BusinessTravel"]:
        df[col] = df[col].astype("string").str.strip()

    # Convert numeric fields safely.
    numeric_columns = SATISFACTION_COLUMNS + [
        "WorkLifeBalance",
        "JobLevel",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "DistanceFromHome",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Satisfaction ratings must be between 1 and 4.
    for col in SATISFACTION_COLUMNS + ["WorkLifeBalance"]:
        df.loc[~df[col].between(1, 4), col] = np.nan

    # Convert Attrition to a consistent binary representation.
    attrition = df["Attrition"].astype("string").str.strip().str.lower()
    df["Attrition"] = attrition.map({
        "1": 1, "yes": 1, "true": 1,
        "0": 0, "no": 0, "false": 0,
    })

    return df


def calculate_metrics(df):
    """Create engagement, workload and burnout diagnostic indicators."""
    df = df.copy()

    # Average the available satisfaction dimensions.
    # Scores are only produced when at least 3 of 4 dimensions exist.
    df["EngagementIndex"] = df[SATISFACTION_COLUMNS].mean(
        axis=1, skipna=True
    ).where(df[SATISFACTION_COLUMNS].notna().sum(axis=1) >= 3)

    df["EngagementScore"] = df["EngagementIndex"] / 4 * 100

    df["EngagementCategory"] = pd.cut(
        df["EngagementScore"],
        bins=[-0.01, 60, 80, 100],
        labels=["Low", "Moderate", "High"],
        include_lowest=True,
    )

    # Work-life balance KPI: 0–100 scale.
    df["WorkLifeBalanceIndex"] = df["WorkLifeBalance"] / 4 * 100

    # Diagnostic risk score: overtime = 50 points;
    # work-life balance <= 2 = 50 points.
    overtime = df["OverTime"].str.lower().eq("yes")
    poor_balance = df["WorkLifeBalance"].le(2)

    df["BurnoutRiskScore"] = (
        overtime.fillna(False).astype(int) * 50
        + poor_balance.fillna(False).astype(int) * 50
    )

    df["BurnoutRiskLevel"] = np.select(
        [
            df["BurnoutRiskScore"].eq(100),
            df["BurnoutRiskScore"].eq(50),
        ],
        ["High", "Medium"],
        default="Low",
    )

    # Workload stress indicator: transparent proxy, not a clinical measure.
    travel_points = df["BusinessTravel"].map({
        "Non-Travel": 0,
        "Travel_Rarely": 1,
        "Travel Rarely": 1,
        "Travel_Frequently": 2,
        "Travel Frequently": 2,
    })

    overtime_points = overtime.astype(int)
    df["WorkloadStressIndicator"] = (
        (travel_points + overtime_points) / 3 * 100
    )

    # Satisfaction stability: lower variation means more consistent ratings.
    satisfaction_std = df[SATISFACTION_COLUMNS].std(
        axis=1, skipna=True, ddof=0
    )
    df["SatisfactionStabilityScore"] = (
        (1 - satisfaction_std / 1.5).clip(0, 1) * 100
    ).where(df[SATISFACTION_COLUMNS].notna().sum(axis=1) >= 2)

    return df


def save_outputs(df):
    """Export the calculated employee-level data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_DIR / "analysis_report.csv", index=False)


def summarize_metrics(df):
    """Return organization-level KPI values."""
    return {
        "Employees": len(df),
        "Average Engagement": df["EngagementScore"].mean(),
        "Low Engagement %": (
            df["EngagementCategory"].eq("Low").mean() * 100
        ),
        "High Burnout Risk": (
            df["BurnoutRiskLevel"].eq("High").sum()
        ),
        "Average Work-Life Balance": (
            df["WorkLifeBalanceIndex"].mean()
        ),
        "Average Satisfaction Stability": (
            df["SatisfactionStabilityScore"].mean()
        ),
        "Average Workload Stress": (
            df["WorkloadStressIndicator"].mean()
        ),
    }
