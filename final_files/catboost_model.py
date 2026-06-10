from preprocessing_final import *
import optuna
import joblib

# IMPORTANT: This code was originally created in appliedAI_finalexperiments.ipynb and was manually converted to a .py file.
# To preserve the functionality from the .ipynb file, some functions need to pass many variables.

def prepare_catboost_data(X_train, X_val, X_trainval, X_test, cat_cols):
    # CatBoost uses the same selected columns
    # it can take the categorical columns without one-hot encoding

    X_cb_train = X_train.copy()
    X_cb_val = X_val.copy()
    X_cb_trainval = X_trainval.copy()
    X_cb_test = X_test.copy()

    for col in cat_cols:
        X_cb_train[col] = X_cb_train[col].fillna("MISSING").astype(str)
        X_cb_val[col] = X_cb_val[col].fillna("MISSING").astype(str)
        X_cb_trainval[col] = X_cb_trainval[col].fillna("MISSING").astype(str)
        X_cb_test[col] = X_cb_test[col].fillna("MISSING").astype(str)

    cat_features = [X_cb_train.columns.get_loc(col) for col in cat_cols]

    return X_cb_train, X_cb_val, X_cb_trainval, X_cb_test, cat_features

def initial_catboost(X_cb_train, y_train, cat_features, X_cb_val, y_val):
    # first CatBoost run before tuning

    cat_plain = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=1,
        verbose=0,
        allow_writing_files=False,
    )

    cat_plain.fit(
        X_cb_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_cb_val, y_val),
        early_stopping_rounds=100,
        verbose=0,
    )

    plain_prob = cat_plain.predict_proba(X_cb_val)[:, 1]
    print("plain catboost val auc:", roc_auc_score(y_val, plain_prob))

def tune_catboost(X_cb_train, y_train, cat_features, X_cb_val, y_val):
    import optuna

    # Optuna tuning for CatBoost
    # this is the slow part, so we print each trial

    # The objective specifies the hyperparameters that Optuna will tune
    def objective(trial):
        # Optuna uses trials, each trial is one experiment using specific hyperparameters
        params = {
            # iterations is the number of boosting trees.
            # Increasing the amount of iterations can increase the learning capacity,
            # but adding too many iterations risks overfitting or long training time.
            "iterations": trial.suggest_int("iterations", 500, 2000),
            # learning_rate controls the speed of the training.
            # Low learning rate trains slowly but might generalize better.
            # High learning rate trains faster but risks overshooting.
            # log=True causes Optuna to search logarithmically, which means it explores
            # lower values more carefully.
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            # depth is the maximum depth of the decision trees.
            # Increasing the depth can allow the model to learn more complex patterns and
            # feature interactions, but increases the risk of overfitting.
            "depth": trial.suggest_int("depth", 4, 8),
            # l2_leaf_reg controls the strength of the L2 regularization.
            # L2 regularization helps reduce overfitting by penalizing large model weights.
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
            # CatBoost automatically chooses the mathematically most optimal split in
            # its decision trees based on the training data, which can cause overfitting.
            # random_strength introduces randomness into the splitting process to reduce
            # overfitting, though if the random_strength is too high model stability can decrease.
            "random_strength": trial.suggest_float("random_strength", 0.001, 10, log=True),
            # CatBoost uses bagging, which weighs some training examples more heavily during
            # each iteration. Increasing the bagging_temperature makes the weights more uneven,
            # which can improve regularization, but can cause the model to become unstable.
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
            # CatBoost creates borders for numeric features and will only split around those borders.
            # Increasing border_count allows for more splits, which can increase accurcacy, but also
            # increases training time.
            "border_count": trial.suggest_categorical("border_count", [32, 64, 128, 254]),
            # Logistic loss function is used because our model has a binary target
            "loss_function": "Logloss",
            # We evaluate on ROC-AUC
            "eval_metric": "AUC",
            # This balances class weights to reduce issues caused by class imbalance
            "auto_class_weights": "Balanced",
            # Fixed seed for reproducibility
            "random_seed": 42,
            # This enables/disables verbose training output
            "verbose": 0,
        }

        print("trial", trial.number + 1, "/", N_TRIALS)

        model = CatBoostClassifier(**params)
        model.fit(
            X_cb_train,
            y_train,
            cat_features=cat_features,
            eval_set=(X_cb_val, y_val),
            early_stopping_rounds=100,
            verbose=0,
        )

        prob = model.predict_proba(X_cb_val)[:, 1]
        auc = roc_auc_score(y_val, prob)

        print("val auc:", round(auc, 4))
        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS)

    print("best val auc:", study.best_value)
    print(study.best_params)

    return study

def train_catboost(study, X_cb_trainval, cat_cols, y_trainval, X_cb_test, y_test):
    # final CatBoost model

    best_params = study.best_params.copy()
    joblib.dump(best_params, "params/best_params.pkl")

    cat_final = CatBoostClassifier(
        **best_params,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=1,
        verbose=100,
        allow_writing_files=False,
    )

    cat_features_final = [X_cb_trainval.columns.get_loc(col) for col in cat_cols]

    cat_final.fit(
        X_cb_trainval,
        y_trainval,
        cat_features=cat_features_final,
    )

    cat_train_prob = cat_final.predict_proba(X_cb_trainval)[:, 1]
    cat_test_prob = cat_final.predict_proba(X_cb_test)[:, 1]

    cat_train_pred = cat_final.predict(X_cb_trainval)
    cat_test_pred = cat_final.predict(X_cb_test)

    print("catboost train auc:", roc_auc_score(y_trainval, cat_train_prob))
    print("catboost test auc:", roc_auc_score(y_test, cat_test_prob))
    print("catboost test f1:", f1_score(y_test, cat_test_pred))
    print("catboost test acc:", accuracy_score(y_test, cat_test_pred))
    print(confusion_matrix(y_test, cat_test_pred))

    imp = pd.DataFrame({
        "feature": X_cb_trainval.columns,
        "importance": cat_final.get_feature_importance(),
    })

    imp = imp.sort_values("importance", ascending=False)
    print(imp.head(15))

    cat_final.save_model("models/catboost_model.cbm")

def main():
    X, y, num_cols, cat_cols = load_and_prepare_data()
    X_train, X_val, X_trainval, X_test, y_train, y_val, y_trainval, y_test = split(X, y)
    X_cb_train, X_cb_val, X_cb_trainval, X_cb_test, cat_features = prepare_catboost_data(X_train, X_val, X_trainval, X_test, cat_cols)
    initial_catboost(X_cb_train, y_train, cat_features, X_cb_val, y_val)
    study = tune_catboost(X_cb_train, y_train, cat_features, X_cb_val, y_val)
    train_catboost(study, X_cb_trainval, cat_cols, y_trainval, X_cb_test, y_test)

if __name__ == "__main__":
    main()