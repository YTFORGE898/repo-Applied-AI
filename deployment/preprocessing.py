import json
import pandas as pd
import numpy as np

def load_metadata(path="models/metadata.json"):
    with open(path, "r") as f:
        return json.load(f)


def preprocess_inference_record(record, metadata):

    data = pd.DataFrame([record])

    # funding

    data["funding_num"] = data["funding_total_usd"].replace("-", np.nan)
    data["funding_num"] = pd.to_numeric(data["funding_num"], errors="coerce")
    data["funding_missing"] = data["funding_num"].isna().astype(int)
    data["log_funding"] = np.log1p(data["funding_num"])
    data["funding_rounds"] = pd.to_numeric(data["funding_rounds"], errors="coerce")

    # dates

    data["founded_at"] = pd.to_datetime(data["founded_at"], errors="coerce")
    data["first_funding_at"] = pd.to_datetime(data["first_funding_at"], errors="coerce")
    data["last_funding_at"] = pd.to_datetime(data["last_funding_at"], errors="coerce")

    # a few dates are clearly broken, so I set those to missing

    for col in ["founded_at", "first_funding_at", "last_funding_at"]:
        bad_date = ((data[col].dt.year < 1900) | (data[col].dt.year > 2016))
        data.loc[bad_date, col] = pd.NaT

    data["founded_missing"] = (data["founded_at"].isna().astype(int))
    data["first_funding_missing"] = (data["first_funding_at"].isna().astype(int))
    data["last_funding_missing"] = (data["last_funding_at"].isna().astype(int))

    data["founded_year"] = data["founded_at"].dt.year
    data["first_year"] = data["first_funding_at"].dt.year
    data["last_year"] = data["last_funding_at"].dt.year

    data["first_gap"] = (data["first_funding_at"] - data["founded_at"]).dt.days
    data["funding_days"] = (data["last_funding_at"] - data["first_funding_at"]).dt.days
    data["startup_days"] = (data["last_funding_at"] - data["founded_at"]).dt.days

    data["bad_first_gap"] = ((data["first_gap"] < 0).fillna(False).astype(int))
    data.loc[data["first_gap"] < 0, "first_gap"] = np.nan
    data["bad_funding_days"] = ((data["funding_days"] < 0).fillna(False).astype(int))
    data.loc[data["funding_days"] < 0, "funding_days"] = np.nan
    data["bad_startup_days"] = ((data["startup_days"] < 0).fillna(False).astype(int))
    data.loc[data["startup_days"] < 0, "startup_days"] = np.nan

    # category count

    data["main_cat"] = (data["category_list"].fillna("MISSING").str.split("|").str[0])
    data["n_cat"] = (data["category_list"].fillna("").str.count(r"\|") + data["category_list"].notna().astype(int))

    # category and location columns
    # use saved training metadata rather than recomputing counts

    data["country_small"] = (data["country_code"].fillna("MISSING"))
    data.loc[~data["country_small"].isin(metadata["keep_country"]), "country_small"] = "OTHER"

    data["main_cat_small"] = (data["main_cat"].fillna("MISSING"))
    data.loc[~data["main_cat_small"].isin(metadata["keep_cat"]), "main_cat_small"] = "OTHER"

    data["state_small"] = (data["state_code"].fillna("MISSING"))
    data.loc[~data["state_small"].isin(metadata["keep_state"]), "state_small"] = "OTHER"

    data["region_small"] = (data["region"].fillna("MISSING"))
    data.loc[~data["region_small"].isin(metadata["keep_region"]), "region_small"] = "OTHER"

    data["city_small"] = (data["city"].fillna("MISSING"))
    data.loc[~data["city_small"].isin(metadata["keep_city"]), "city_small"] = "OTHER"

    # final features
    # use the exact feature order from training

    X = data[metadata["feature_order"]].copy()

    return X