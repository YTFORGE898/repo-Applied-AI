import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

# IMPORTANT: This code was originally created in appliedAI_finalexperiments.ipynb and was manually converted to a .py file.
# To preserve the functionality from the .ipynb file, some functions need to pass many variables.

# settings for this experiment

country_min_count = 10
main_cat_min_count = 20
state_min_count = 20
region_min_count = 20
city_min_count = 30
C_values = [0.01, 0.1, 1, 10, 100]
N_TRIALS = 30
TIMEOUT_SECONDS = 1800

def load_and_prepare_data(csv_path = "big_startup_secsees_dataset.csv"):
    df = pd.read_csv(csv_path)
    print("raw shape:", df.shape)
    df.head()

    # same target as the first baseline

    good_status = ["closed", "acquired", "ipo"]
    data = df[df["status"].isin(good_status)].copy()
    data["target"] = 0
    data.loc[data["status"].isin(["acquired", "ipo"]), "target"] = 1

    print("labelled shape:", data.shape)
    print(data["target"].value_counts())

    # duplicate checks

    print("duplicate rows:", data.duplicated().sum())
    print("duplicate permalinks:", data["permalink"].duplicated().sum())

    name_check = data["name"].fillna("").str.lower().str.strip()
    home_check = data["homepage_url"].fillna("").str.lower().str.strip()

    print("duplicate names:", name_check[name_check != ""].duplicated().sum())
    print("duplicate homepages:", home_check[home_check != ""].duplicated().sum())

    # funding

    data["funding_num"] = data["funding_total_usd"].replace("-", np.nan)
    data["funding_num"] = pd.to_numeric(data["funding_num"], errors="coerce")
    data["funding_missing"] = data["funding_num"].isna().astype(int)
    data["log_funding"] = np.log1p(data["funding_num"]) #some values are zero, so we use log1p to avoid -inf
    data["funding_rounds"] = pd.to_numeric(data["funding_rounds"], errors="coerce")

    # dates

    data["founded_at"] = pd.to_datetime(data["founded_at"], errors="coerce")
    data["first_funding_at"] = pd.to_datetime(data["first_funding_at"], errors="coerce")
    data["last_funding_at"] = pd.to_datetime(data["last_funding_at"], errors="coerce")

    # a few dates are clearly broken, so I set those to missing
    for col in ["founded_at", "first_funding_at", "last_funding_at"]:
        bad_date = (data[col].dt.year < 1900) | (data[col].dt.year > 2016) #1900 is arbitrary, startups in the dataset shouldnt be founded before that, 2016 is the last year of data collection
        data.loc[bad_date, col] = pd.NaT

    data["founded_missing"] = data["founded_at"].isna().astype(int)
    data["first_funding_missing"] = data["first_funding_at"].isna().astype(int)
    data["last_funding_missing"] = data["last_funding_at"].isna().astype(int)

    data["founded_year"] = data["founded_at"].dt.year
    data["first_year"] = data["first_funding_at"].dt.year
    data["last_year"] = data["last_funding_at"].dt.year

    data["first_gap"] = (data["first_funding_at"] - data["founded_at"]).dt.days
    data["funding_days"] = (data["last_funding_at"] - data["first_funding_at"]).dt.days
    data["startup_days"] = (data["last_funding_at"] - data["founded_at"]).dt.days

    data["bad_first_gap"] = (data["first_gap"] < 0).fillna(False).astype(int)
    data.loc[data["first_gap"] < 0, "first_gap"] = np.nan
    data["bad_funding_days"] = (data["funding_days"] < 0).fillna(False).astype(int)
    data.loc[data["funding_days"] < 0, "funding_days"] = np.nan
    data["bad_startup_days"] = (data["startup_days"] < 0).fillna(False).astype(int)
    data.loc[data["startup_days"] < 0, "startup_days"] = np.nan

    # category count is needed for the correlation check

    data["main_cat"] = data["category_list"].fillna("MISSING").str.split("|").str[0]
    data["n_cat"] = data["category_list"].fillna("").str.count(r"\|") + data["category_list"].notna().astype(int)

    # Spearman correlation before choosing the final date columns

    corr_cols = [
        "funding_rounds",
        "log_funding",
        "founded_year",
        "first_year",
        "last_year",
        "first_gap",
        "funding_days",
        "startup_days",
        "n_cat",
    ]

    corr = data[corr_cols].corr(method="spearman")
    print(corr.round(3))

    pairs = []
    for i in range(len(corr_cols)):
        for j in range(i + 1, len(corr_cols)):
            value = corr.iloc[i, j]
            if abs(value) >= 0.8:
                pairs.append([corr_cols[i], corr_cols[j], value])

    high_corr = pd.DataFrame(pairs, columns=["var1", "var2", "spearman"])
    high_corr

    # category and location columns
    # this is the bigger feature version compared with the first baseline

    data["country_small"] = data["country_code"].fillna("MISSING")
    country_counts = data["country_small"].value_counts()
    keep_country = country_counts[country_counts >= country_min_count].index
    data.loc[~data["country_small"].isin(keep_country), "country_small"] = "OTHER"

    data["main_cat_small"] = data["main_cat"].fillna("MISSING")
    cat_counts = data["main_cat_small"].value_counts()
    keep_cat = cat_counts[cat_counts >= main_cat_min_count].index
    data.loc[~data["main_cat_small"].isin(keep_cat), "main_cat_small"] = "OTHER"

    data["state_small"] = data["state_code"].fillna("MISSING")
    state_counts = data["state_small"].value_counts()
    keep_state = state_counts[state_counts >= state_min_count].index
    data.loc[~data["state_small"].isin(keep_state), "state_small"] = "OTHER"

    data["region_small"] = data["region"].fillna("MISSING")
    region_counts = data["region_small"].value_counts()
    keep_region = region_counts[region_counts >= region_min_count].index
    data.loc[~data["region_small"].isin(keep_region), "region_small"] = "OTHER"

    data["city_small"] = data["city"].fillna("MISSING")
    city_counts = data["city_small"].value_counts()
    keep_city = city_counts[city_counts >= city_min_count].index
    data.loc[~data["city_small"].isin(keep_city), "city_small"] = "OTHER"

    print("country levels:", data["country_small"].nunique())
    print("category levels:", data["main_cat_small"].nunique())
    print("state levels:", data["state_small"].nunique())
    print("region levels:", data["region_small"].nunique())
    print("city levels:", data["city_small"].nunique())

    # final features for experiment B
    # last_funding_at based columns are not used here

    num_cols = [
        "funding_rounds",
        "log_funding",
        "funding_missing",
        "founded_year",
        "first_year",
        "first_gap",
        "founded_missing",
        "first_funding_missing",
        "bad_first_gap",
        "n_cat",
    ]

    cat_cols = [
        "country_small",
        "main_cat_small",
        "state_small",
        "region_small",
        "city_small",
    ]

    X = data[num_cols + cat_cols].copy()
    y = data["target"].copy()

    print("raw columns:", X.shape[1])
    print("numeric:", len(num_cols), "categorical:", len(cat_cols))

    return X, y, num_cols, cat_cols

def split(X, y):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=1,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        stratify=y_trainval,
        random_state=1,
    )

    print("train:", X_train.shape)
    print("val:", X_val.shape)
    print("test:", X_test.shape)

    return X_train, X_val, X_trainval, X_test, y_train, y_val, y_trainval, y_test

def logistic_regression_preprocessing(num_cols, cat_cols):
    # same preprocessing for the logistic baseline

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    prep = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    return prep