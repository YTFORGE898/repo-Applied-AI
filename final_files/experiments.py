from preprocessing_final import *
from logistic_regression import *
from catboost_model import *
# more stuff to import
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss

# IMPORTANT: This code was originally created in appliedAI_finalexperiments.ipynb and was manually converted to a .py file.
# To preserve the functionality from the .ipynb file, some functions need to pass many variables.
# Make sure to run logistic_regression.py and catboost_model.py before running this file.
# The models and some parameters will be saved when running the files above, which will be loaded and used in this file.

out_path = "presentation_experiment_outputs"
os.makedirs(out_path, exist_ok=True)

def experiment_one_split(y_test, base_test_prob, base_test_pred, cat_test_prob, cat_test_pred, y_trainval, base_train_prob, base_train_pred, cat_train_prob, cat_train_pred, X_trainval, cat_cols, best_params):
    rows = []

    rows.append({
        "model": "logistic baseline",
        "test_auc": roc_auc_score(y_test, base_test_prob),
        "test_f1": f1_score(y_test, base_test_pred),
        "test_acc": accuracy_score(y_test, base_test_pred),
    })

    rows.append({
        "model": "catboost tuned",
        "test_auc": roc_auc_score(y_test, cat_test_prob),
        "test_f1": f1_score(y_test, cat_test_pred),
        "test_acc": accuracy_score(y_test, cat_test_pred),
    })

    df_split = pd.DataFrame(rows)
    df_split["auc_minus_baseline"] = df_split["test_auc"] - df_split.loc[0, "test_auc"]
    df_split.to_csv(os.path.join(out_path, "one_split_test_result.csv"), index=False)
    print(df_split)

    # check the train-test gap

    rows = []

    rows.append({
        "model": "logistic baseline",
        "train_auc": roc_auc_score(y_trainval, base_train_prob),
        "test_auc": roc_auc_score(y_test, base_test_prob),
        "train_f1": f1_score(y_trainval, base_train_pred),
        "test_f1": f1_score(y_test, base_test_pred),
        "train_acc": accuracy_score(y_trainval, base_train_pred),
        "test_acc": accuracy_score(y_test, base_test_pred),
    })

    rows.append({
        "model": "catboost tuned",
        "train_auc": roc_auc_score(y_trainval, cat_train_prob),
        "test_auc": roc_auc_score(y_test, cat_test_prob),
        "train_f1": f1_score(y_trainval, cat_train_pred),
        "test_f1": f1_score(y_test, cat_test_pred),
        "train_acc": accuracy_score(y_trainval, cat_train_pred),
        "test_acc": accuracy_score(y_test, cat_test_pred),
    })

    check_df = pd.DataFrame(rows)
    check_df["auc_gap"] = check_df["train_auc"] - check_df["test_auc"]
    check_df["f1_gap"] = check_df["train_f1"] - check_df["test_f1"]
    check_df["acc_gap"] = check_df["train_acc"] - check_df["test_acc"]

    notes = []
    for _, row in check_df.iterrows():
        if row["train_auc"] < 0.70 and row["test_auc"] < 0.70:
            notes.append("possible underfitting")
        elif row["auc_gap"] > 0.1:
            notes.append("possible overfitting")
        else:
            notes.append("no large train-test gap")

    check_df["rough_check"] = notes
    check_df.to_csv(os.path.join(out_path, "one_split_fit_check.csv"), index=False)
    check_df.round(4)

    # plot learning curve

    X_lc_train, X_lc_val, y_lc_train, y_lc_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.2,
        stratify=y_trainval,
        random_state=1,
    )

    sizes = [0.2, 0.4, 0.6, 0.8, 1]
    rows = []

    for size in sizes:
        print("training size:", size)

        if size < 1:
            X_part, _, y_part, _ = train_test_split(
                X_lc_train,
                y_lc_train,
                train_size=size,
                stratify=y_lc_train,
                random_state=1,
            )
        else:
            X_part = X_lc_train.copy()
            y_part = y_lc_train.copy()

        X_part_cb = X_part.copy()
        X_lc_val_cb = X_lc_val.copy()

        for col in cat_cols:
            X_part_cb[col] = X_part_cb[col].fillna("MISSING").astype(str)
            X_lc_val_cb[col] = X_lc_val_cb[col].fillna("MISSING").astype(str)

        cat_features = [X_part_cb.columns.get_loc(col) for col in cat_cols]

        lc_params = best_params.copy()
        lc_params["iterations"] = 500
        lc_params["loss_function"] = "Logloss"
        lc_params["eval_metric"] = "AUC"
        lc_params["auto_class_weights"] = "Balanced"
        lc_params["random_seed"] = 1
        lc_params["verbose"] = 0
        lc_params["allow_writing_files"] = False

        model = CatBoostClassifier(**lc_params)

        model.fit(
            X_part_cb,
            y_part,
            cat_features=cat_features,
            eval_set=(X_lc_val_cb, y_lc_val),
            early_stopping_rounds=50,
            verbose=0,
        )

        train_prob = model.predict_proba(X_part_cb)[:, 1]
        val_prob = model.predict_proba(X_lc_val_cb)[:, 1]

        rows.append({
            "train_fraction": size,
            "train_rows": len(X_part_cb),
            "train_auc": roc_auc_score(y_part, train_prob),
            "val_auc": roc_auc_score(y_lc_val, val_prob),
        })

    learn_df = pd.DataFrame(rows)
    learn_df["auc_gap"] = learn_df["train_auc"] - learn_df["val_auc"]

    learn_df.to_csv(
        os.path.join(out_path, "learning_curve.csv"),
        index=False,
    )

    learn_df.round(4)

    plt.figure(figsize=(7, 5))

    plt.plot(
        learn_df["train_rows"],
        learn_df["train_auc"],
        marker="o",
        label="train AUC",
    )

    plt.plot(
        learn_df["train_rows"],
        learn_df["val_auc"],
        marker="o",
        label="validation AUC",
    )

    plt.xlabel("training rows")
    plt.ylabel("AUC")
    plt.title("CatBoost learning curve")
    plt.legend()
    plt.grid()

    plt.savefig(
        os.path.join(out_path, "learning_curve.png"),
        bbox_inches="tight",
    )
    plt.savefig(os.path.join(out_path, "learning_curve.png"), dpi=300, bbox_inches="tight")
    plt.show()

def experiment_cross_validation(best_params, X_trainval, y_trainval, prep, best_C, X_test, cat_cols):
    cv_folds = 5
    cv_stop = 50
    cv_params = best_params.copy()
    cv_params["loss_function"] = "Logloss"
    cv_params["eval_metric"] = "AUC"
    cv_params["auto_class_weights"] = "Balanced"
    cv_params["verbose"] = 0
    cv_params["allow_writing_files"] = False

    # convenience functions on metrics

    def score_row(model_name, fold, y_true, prob):
        pred = (prob >= 0.5).astype(int)

        return {
            "model": model_name,
            "fold": fold,
            "auc": roc_auc_score(y_true, prob),
            "f1": f1_score(y_true, pred),
            "acc": accuracy_score(y_true, pred),
            "brier": brier_score_loss(y_true, prob),
            "logloss": log_loss(y_true, prob, labels=[0, 1]),
        }

    def make_cv_summary(res):
        rows = []
        metrics = ["auc", "f1", "acc", "brier", "logloss"]

        for model in res["model"].unique():
            part = res[res["model"] == model]

            for metric in metrics:
                vals = part[metric].values
                std = vals.std(ddof=1)

                rows.append({
                    "model": model,
                    "metric": metric,
                    "mean": vals.mean(),
                    "std": std,
                    "ci95": 1.96 * std / np.sqrt(len(vals)),
                })

        return pd.DataFrame(rows)
    
    # running cv

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=1)
    rows = []
    cat_models = []
    test_probs = []
    test_logits = []

    for fold, (train_index, val_index) in enumerate(cv.split(X_trainval, y_trainval), start=1):
        print("fold", fold, "of", cv_folds)

        X_tr = X_trainval.iloc[train_index].copy()
        X_va = X_trainval.iloc[val_index].copy()
        y_tr = y_trainval.iloc[train_index].copy()
        y_va = y_trainval.iloc[val_index].copy()

        # logistic
        base_cv = Pipeline([
            ("prep", prep),
            ("model", LogisticRegression(C=best_C, max_iter=1000, solver="liblinear")),
        ])

        base_cv.fit(X_tr, y_tr)
        base_prob = base_cv.predict_proba(X_va)[:, 1]
        rows.append(score_row("logistic", fold, y_va, base_prob))

        # CatBoost
        X_tr_cb = X_tr.copy()
        X_va_cb = X_va.copy()
        X_test_cb = X_test.copy()

        for col in cat_cols:
            X_tr_cb[col] = X_tr_cb[col].fillna("MISSING").astype(str)
            X_va_cb[col] = X_va_cb[col].fillna("MISSING").astype(str)
            X_test_cb[col] = X_test_cb[col].fillna("MISSING").astype(str)

        cat_features = [X_tr_cb.columns.get_loc(col) for col in cat_cols]

        params = cv_params.copy()
        params["random_seed"] = 1

        cat_cv = CatBoostClassifier(**params)
        cat_cv.fit(
            X_tr_cb,
            y_tr,
            cat_features=cat_features,
            eval_set=(X_va_cb, y_va),
            early_stopping_rounds=cv_stop,
            verbose=0,
        )

        cat_prob = cat_cv.predict_proba(X_va_cb)[:, 1]
        rows.append(score_row("catboost", fold, y_va, cat_prob))

        # save for uncertainty
        test_prob = cat_cv.predict_proba(X_test_cb)[:, 1]
        test_logit = cat_cv.predict(X_test_cb, prediction_type="RawFormulaVal")
        test_logit = np.ravel(test_logit)

        test_probs.append(test_prob)
        test_logits.append(test_logit)
        cat_models.append(cat_cv)

    cv_df = pd.DataFrame(rows)
    cv_table = make_cv_summary(cv_df)
    print(cv_df)

    # cv summary

    cv_mean = cv_table.pivot(index="metric", columns="model", values="mean")
    cv_mean.to_csv(os.path.join(out_path, "cv_summary_mean_wide.csv"))

    print("mean scores")
    print(cv_mean.round(4))

    return test_probs, test_logits, cv_params, score_row

def experiment_uncertainty(test_probs, test_logits, y_test):
    # uncertainty from the five fold models

    prob_arr = np.array(test_probs)
    logit_arr = np.array(test_logits)

    avg_prob = prob_arr.mean(axis=0)
    std_prob = prob_arr.std(axis=0)

    mean_logit = logit_arr.mean(axis=0)
    std_logit = logit_arr.std(axis=0)

    avg_pred = (avg_prob >= 0.5).astype(int)
    correct = (avg_pred == y_test.values).astype(int)

    unc_test = pd.DataFrame({
        "true": y_test.values,
        "pred": avg_pred,
        "mean_prob": avg_prob,
        "std_prob": std_prob,
        "mean_logit": mean_logit,
        "std_logit": std_logit,
        "correct": correct,
    }, index=y_test.index)

    cv_test_df = pd.DataFrame([{
        "model": "catboost cv mean",
        "test_auc": roc_auc_score(y_test, avg_prob),
        "test_f1": f1_score(y_test, avg_pred),
        "test_acc": accuracy_score(y_test, avg_pred),
        "test_brier": brier_score_loss(y_test, avg_prob),
        "test_logloss": log_loss(y_test, avg_prob, labels=[0, 1]),
    }])

    rows = []

    for col in ["std_prob", "std_logit"]:
        rows.append({
            "quantity": col,
            "mean": unc_test[col].mean(),
            "median": unc_test[col].median(),
            "q75": unc_test[col].quantile(0.75),
            "q90": unc_test[col].quantile(0.9),
            "q95": unc_test[col].quantile(0.95),
            "max": unc_test[col].max(),
        })

    unc_table = pd.DataFrame(rows)

    top_unc = unc_test.sort_values("std_logit", ascending=False).head(5)

    cv_test_df.to_csv(os.path.join(out_path, "catboost_cv_ensemble_test_result.csv"), index=False)
    unc_table.to_csv(os.path.join(out_path, "uncertainty_summary.csv"), index=False)
    top_unc.to_csv(os.path.join(out_path, "uncertain_examples.csv"))

    print("CatBoost CV mean result")
    print(cv_test_df.round(4))

    print("uncertainty summary")
    print(unc_table.round(4))

    print("largest fold disagreeing rows")
    print(top_unc[[
        "true",
        "pred",
        "mean_prob",
        "std_prob",
        "mean_logit",
        "std_logit",
        "correct",
    ]].round(4))

    return avg_prob

def experiment_calibration(avg_prob, y_test):
    rows = []
    bins = np.linspace(0, 1, 11)
    bin_no = np.digitize(avg_prob, bins[1:-1], right=True)

    for b in range(10):
        mask = bin_no == b

        if mask.sum() > 0:
            mean_prob = avg_prob[mask].mean()
            obs_rate = y_test.values[mask].mean()
            err = abs(mean_prob - obs_rate)

            rows.append({
                "bin": b,
                "left": bins[b],
                "right": bins[b + 1],
                "count": mask.sum(),
                "mean_predicted_prob": mean_prob,
                "observed_success_rate": obs_rate,
                "abs_error": err,
                "weighted_abs_error": err * mask.sum() / len(y_test),
            })

    calib_table = pd.DataFrame(rows)
    ece = calib_table["weighted_abs_error"].sum()

    calib_table.to_csv(os.path.join(out_path, "calibration_table.csv"), index=False)
    print("ECE:", round(ece, 4))
    print(calib_table)

    # reliability plot

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    plt.plot(
        calib_table["mean_predicted_prob"],
        calib_table["observed_success_rate"],
        marker="o",
        label="CatBoost CV ensemble",
    )
    plt.xlabel("mean predicted success probability")
    plt.ylabel("observed success rate")
    plt.title("Reliability plot")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "reliability_plot.png"), dpi=200)
    plt.show()

def experiment_feature_probing(num_cols, cat_cols, cv_params, X_trainval, y_trainval, best_C, score_row):
    # some trial settings

    exp_folds = 3
    exp_iter = 500
    exp_stop = 50

    feat_sets = [
        {
            "name": "all_features",
            "num": num_cols,
            "cat": cat_cols,
        },
        {
            "name": "no_city",
            "num": num_cols,
            "cat": ["country_small", "main_cat_small", "state_small", "region_small"],
        },
        {
            "name": "geography_only",
            "num": [],
            "cat": ["country_small", "state_small", "region_small", "city_small"],
        },
        {
            "name": "no_funding_amount_or_rounds",
            "num": [c for c in num_cols if c not in ["funding_rounds", "log_funding", "funding_missing"]],
            "cat": cat_cols,
        },
        {
            "name": "early_basic_features",
            "num": ["founded_year", "founded_missing", "n_cat"],
            "cat": ["country_small", "main_cat_small", "state_small"],
        },
    ]

    exp_params = cv_params.copy()
    exp_params["iterations"] = min(int(exp_params.get("iterations", exp_iter)), exp_iter)
    exp_params["verbose"] = 0

    print(exp_params["iterations"])

    # run

    exp_cv = StratifiedKFold(n_splits=exp_folds, shuffle=True, random_state=1)
    rows = []

    for feat_set in feat_sets:
        print("feature set:", feat_set["name"])

        use_num = feat_set["num"]
        use_cat = feat_set["cat"]
        use_cols = use_num + use_cat

        X_exp = X_trainval[use_cols].copy()
        y_exp = y_trainval.copy()

        for fold, (train_index, val_index) in enumerate(exp_cv.split(X_exp, y_exp), start=1):
            X_tr = X_exp.iloc[train_index].copy()
            X_va = X_exp.iloc[val_index].copy()
            y_tr = y_exp.iloc[train_index].copy()
            y_va = y_exp.iloc[val_index].copy()

            # logistic
            num_pipe_exp = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ])

            cat_pipe_exp = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ])

            prep_exp = ColumnTransformer([
                ("num", num_pipe_exp, use_num),
                ("cat", cat_pipe_exp, use_cat),
            ])

            base_exp = Pipeline([
                ("prep", prep_exp),
                ("model", LogisticRegression(C=best_C, max_iter=1000, solver="liblinear")),
            ])

            base_exp.fit(X_tr, y_tr)
            base_prob = base_exp.predict_proba(X_va)[:, 1]
            row = score_row("logistic", fold, y_va, base_prob)
            row["feature_set"] = feat_set["name"]
            rows.append(row)

            # CatBoost
            for col in use_cat:
                X_tr[col] = X_tr[col].fillna("MISSING").astype(str)
                X_va[col] = X_va[col].fillna("MISSING").astype(str)

            cat_features = [X_tr.columns.get_loc(col) for col in use_cat]

            params = exp_params.copy()
            params["random_seed"] = 1

            cat_exp = CatBoostClassifier(**params)
            cat_exp.fit(
                X_tr,
                y_tr,
                cat_features=cat_features,
                eval_set=(X_va, y_va),
                early_stopping_rounds=exp_stop,
                verbose=0,
            )

            cat_prob = cat_exp.predict_proba(X_va)[:, 1]
            row = score_row("catboost", fold, y_va, cat_prob)
            row["feature_set"] = feat_set["name"]
            rows.append(row)

    feat_df = pd.DataFrame(rows)
    feat_df.head()

    # summary table

    rows = []

    for feat_name in feat_df["feature_set"].unique():
        for model in feat_df["model"].unique():
            part = feat_df[
                (feat_df["feature_set"] == feat_name)
                & (feat_df["model"] == model)
            ]

            rows.append({
                "feature_set": feat_name,
                "model": model,
                "auc_mean": part["auc"].mean(),
                "auc_std": part["auc"].std(ddof=1),
                "f1_mean": part["f1"].mean(),
                "acc_mean": part["acc"].mean(),
                "brier_mean": part["brier"].mean(),
            })

    feat_table = pd.DataFrame(rows)
    feat_table = feat_table.sort_values(["feature_set", "model"])
    feat_table.to_csv(os.path.join(out_path, "feature_set_summary.csv"), index=False)
    feat_table.round(4)

def main():
    lr_model = joblib.load("models/logistic_regression.pkl")
    cb_model = CatBoostClassifier()
    cb_model.load_model("models/catboost_model.cbm")
    best_C = joblib.load("params/best_C.pkl")
    best_params = joblib.load("params/best_params.pkl")
    X, y, num_cols, cat_cols = load_and_prepare_data()
    X_train, X_val, X_trainval, X_test, y_train, y_val, y_trainval, y_test = split(X, y)
    prep = logistic_regression_preprocessing(num_cols, cat_cols)
    base_train_prob = lr_model.predict_proba(X_trainval)[:, 1]
    base_test_prob = lr_model.predict_proba(X_test)[:, 1]
    base_train_pred = lr_model.predict(X_trainval)
    base_test_pred = lr_model.predict(X_test)
    X_test_cb = X_test.copy()
    X_trainval_cb = X_trainval.copy()
    for col in cat_cols:
        X_test_cb[col] = X_test_cb[col].fillna("MISSING").astype(str)
        X_trainval_cb[col] = X_trainval_cb[col].fillna("MISSING").astype(str)
    cat_train_prob = cb_model.predict_proba(X_trainval_cb)[:, 1]
    cat_test_prob = cb_model.predict_proba(X_test_cb)[:, 1]
    cat_train_pred = cb_model.predict(X_trainval_cb)
    cat_test_pred = cb_model.predict(X_test_cb)
    experiment_one_split(y_test, base_test_prob, base_test_pred, cat_test_prob, cat_test_pred, y_trainval, base_train_prob, base_train_pred, cat_train_prob, cat_train_pred, X_trainval, cat_cols, best_params)
    test_probs, test_logits, cv_params, score_row = experiment_cross_validation(best_params, X_trainval, y_trainval, prep, best_C, X_test, cat_cols)
    avg_prob = experiment_uncertainty(test_probs, test_logits, y_test)
    experiment_calibration(avg_prob, y_test)
    experiment_feature_probing(num_cols, cat_cols, cv_params, X_trainval, y_trainval, best_C, score_row)

if __name__ == "__main__":
    main()