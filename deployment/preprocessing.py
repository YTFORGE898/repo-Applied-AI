import json
import pandas as pd
import numpy as np


def load_metadata(path):

    with open(path, "r") as f:
        return json.load(f)


def preprocess_inference_record(record, metadata):

    df = pd.DataFrame([record])

    # funding

    df["funding_total_usd"] = (
        df["funding_total_usd"]
        .replace("-", np.nan)
    )

    df["funding_total_usd"] = pd.to_numeric(
        df["funding_total_usd"],
        errors="coerce"
    )

    df["funding_missing"] = (
        df["funding_total_usd"]
        .isna()
        .astype(int)
    )

    df["funding_total_usd"] = (
        df["funding_total_usd"]
        .fillna(0)
    )

    df["funding_total_usd_log"] = np.log1p(
        df["funding_total_usd"]
    )

    # funding rounds

    df["funding_rounds"] = pd.to_numeric(
        df["funding_rounds"],
        errors="coerce"
    )

    df["funding_rounds"] = (
        df["funding_rounds"]
        .fillna(0)
    )

    # dates

    date_cols = [
        "founded_at",
        "first_funding_at",
        "last_funding_at"
    ]

    for col in date_cols:

        df[col] = pd.to_datetime(
            df[col],
            errors="coerce"
        )

        bad_date = (
            (df[col].dt.year < 1900)
            |
            (df[col].dt.year > 2016)
        )

        df.loc[bad_date, col] = pd.NaT

    df["founded_at_missing"] = (
        df["founded_at"].isna().astype(int)
    )

    df["first_funding_at_missing"] = (
        df["first_funding_at"].isna().astype(int)
    )

    df["last_funding_at_missing"] = (
        df["last_funding_at"].isna().astype(int)
    )

    reference_date = pd.Timestamp(
        metadata["reference_date"]
    )

    df["startup_age_days"] = (
        reference_date - df["founded_at"]
    ).dt.days

    df["time_to_first_funding_days"] = (
        df["first_funding_at"] - df["founded_at"]
    ).dt.days

    df["funding_duration_days"] = (
        df["last_funding_at"] - df["first_funding_at"]
    ).dt.days

    for col in [
        "startup_age_days",
        "time_to_first_funding_days",
        "funding_duration_days"
    ]:

        df[col] = df[col].fillna(0)

    # category features

    if "main_category" not in df.columns:

        df["main_category"] = (
            df["category_list"]
            .fillna("missing")
            .str.split("|")
            .str[0]
        )

    df["category_count"] = 1

    if "category_list" in df.columns:

        df["category_count"] = (
            df["category_list"]
            .fillna("")
            .str.count(r"\|")
            + df["category_list"].notna().astype(int)
        )

    # categorical columns

    cat_cols = [
        "main_category",
        "country_code",
        "state_code",
        "region",
        "city"
    ]

    for col in cat_cols:

        df[col] = (
            df[col]
            .fillna("missing")
            .astype(str)
        )

    # date decomposition

    df["founded_at_year"] = (
        df["founded_at"]
        .dt.year
        .fillna(0)
        .astype(int)
    )

    df["founded_at_month"] = (
        df["founded_at"]
        .dt.month
        .fillna(0)
        .astype(int)
    )

    df["first_funding_at_year"] = (
        df["first_funding_at"]
        .dt.year
        .fillna(0)
        .astype(int)
    )

    df["first_funding_at_month"] = (
        df["first_funding_at"]
        .dt.month
        .fillna(0)
        .astype(int)
    )

    df["last_funding_at_year"] = (
        df["last_funding_at"]
        .dt.year
        .fillna(0)
        .astype(int)
    )

    df["last_funding_at_month"] = (
        df["last_funding_at"]
        .dt.month
        .fillna(0)
        .astype(int)
    )

    # exact feature order used during training

    df = df[
        metadata["feature_order"]
    ]

    return df