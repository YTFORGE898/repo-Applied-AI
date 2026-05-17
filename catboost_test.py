import pandas as pd
import numpy as np
import time
import optuna

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# run preprocessing file first

df = pd.read_csv("startup_preprocessed.csv")

target = "target"
cat_cols = ["main_category", "country_code", "state_code", "region", "city"]
date_cols = ["founded_at", "first_funding_at", "last_funding_at"]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df[col + "_year"] = df[col].dt.year
    df[col + "_month"] = df[col].dt.month

df = df.drop(columns=date_cols)

for col in cat_cols:
    df[col] = df[col].fillna("missing").astype(str)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove(target)

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

X = df.drop(columns=[target])
y = df[target]

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.25,
    stratify=y_trainval,
    random_state=42,
)

cat_features = [X.columns.get_loc(col) for col in cat_cols]

# first plain CatBoost model

cat0 = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=0,
)

start = time.time()
cat0.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    early_stopping_rounds=100,
    verbose=0,
)
plain_time = time.time() - start

val_prob = cat0.predict_proba(X_val)[:, 1]
print("plain catboost val auc:", roc_auc_score(y_val, val_prob))
print("plain catboost time:", round(plain_time, 2))

# we use Optuna to search for the best parameters

N_TRIALS = 30
TIMEOUT_SECONDS = 1800

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.001, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128, 254]),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "verbose": 0,
    }

    print("trial", trial.number + 1, "/", N_TRIALS)

    model = CatBoostClassifier(**params)
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=0,
    )

    prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, prob)

    print("val auc:", round(auc, 4))
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS)

print("best val auc:", study.best_value)
print(study.best_params)

best_params = study.best_params.copy()

cat_final = CatBoostClassifier(
    **best_params,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=100,
)

cat_final.fit(
    X_trainval,
    y_trainval,
    cat_features=cat_features,
)

train_prob = cat_final.predict_proba(X_trainval)[:, 1]
test_prob = cat_final.predict_proba(X_test)[:, 1]

train_pred = cat_final.predict(X_trainval)
test_pred = cat_final.predict(X_test)

print("train auc:", roc_auc_score(y_trainval, train_prob))
print("test auc:", roc_auc_score(y_test, test_prob))
print("test f1:", f1_score(y_test, test_pred))
print("test acc:", accuracy_score(y_test, test_pred))
print(confusion_matrix(y_test, test_pred))

imp = pd.DataFrame({
    "feature": X.columns,
    "importance": cat_final.get_feature_importance(),
})

imp = imp.sort_values("importance", ascending=False)
print(imp.head(15))
