import pandas as pd
import numpy as np

# Initial version of the preprocessing. The final version is in the final_files folder.
# This version is used for deployment.

df = pd.read_csv("big_startup_secsees_dataset.csv")

success_labels = ["acquired", "ipo"]
failure_labels = ["closed"]

df = df[df["status"].isin(success_labels + failure_labels)].copy()

df["target"] = 0
df.loc[df["status"].isin(success_labels), "target"] = 1

print("rows after filtering:", df.shape)
print(df["target"].value_counts())

df = df.drop(columns=["permalink", "name", "homepage_url", "status"])

df["funding_total_usd"] = df["funding_total_usd"].replace("-", np.nan)
df["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"], errors="coerce")

df["funding_missing"] = df["funding_total_usd"].isna().astype(int)
df["funding_total_usd"] = df["funding_total_usd"].fillna(df["funding_total_usd"].median())
df["funding_total_usd_log"] = np.log1p(df["funding_total_usd"])

df["funding_rounds"] = pd.to_numeric(df["funding_rounds"], errors="coerce")
df["funding_rounds"] = df["funding_rounds"].fillna(df["funding_rounds"].median())

date_cols = ["founded_at", "first_funding_at", "last_funding_at"]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    bad_date = (df[col].dt.year < 1900) | (df[col].dt.year > 2016)
    df.loc[bad_date, col] = pd.NaT
    df[col + "_missing"] = df[col].isna().astype(int)

reference_date = df["last_funding_at"].max()

df["startup_age_days"] = (reference_date - df["founded_at"]).dt.days
df["time_to_first_funding_days"] = (df["first_funding_at"] - df["founded_at"]).dt.days
df["funding_duration_days"] = (df["last_funding_at"] - df["first_funding_at"]).dt.days

for col in ["startup_age_days", "time_to_first_funding_days", "funding_duration_days"]:
    df[col] = df[col].fillna(df[col].median())

df["main_category"] = df["category_list"].fillna("missing").str.split("|").str[0]
df["category_count"] = df["category_list"].fillna("").str.count(r"\|") + df["category_list"].notna().astype(int)

df = df.drop(columns=["category_list"])

cat_cols = ["main_category", "country_code", "state_code", "region", "city"]

for col in cat_cols:
    df[col] = df[col].fillna("missing").astype(str)

output_file = "startup_preprocessed.csv"
df.to_csv(output_file, index=False)

print("saved:", output_file)
print(df.shape)
print(df.head())
