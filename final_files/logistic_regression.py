from preprocessing_final import *
import joblib
import os

# IMPORTANT: This code was originally created in appliedAI_finalexperiments.ipynb and was manually converted to a .py file.
# To preserve the functionality from the .ipynb file, some functions need to pass many variables.

def tune_logistic_regression(X_train, X_val, y_train, y_val, prep):
    # logistic regression baseline
    # this is fast, so I just print the small table

    rows = []

    for C in C_values:
        model = Pipeline([
            ("prep", prep),
            ("model", LogisticRegression(C=C, max_iter=1000, solver="liblinear")),
        ])

        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]
        pred = model.predict(X_val)

        rows.append({
            "C": C,
            "val_auc": roc_auc_score(y_val, prob),
            "val_f1": f1_score(y_val, pred),
            "val_acc": accuracy_score(y_val, pred),
        })

    base_res = pd.DataFrame(rows)
    print(base_res)

    best_C = base_res.sort_values("val_auc", ascending=False).iloc[0]["C"]
    print("best C:", best_C)

    os.makedirs("params", exist_ok=True)
    joblib.dump(best_C, "params/best_C.pkl")

    return best_C, base_res

def train_logistic_regression(X_trainval, y_trainval, X_test, y_test, prep, best_C):
    base_model = Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(C=best_C, max_iter=1000, solver="liblinear")),
    ])

    base_model.fit(X_trainval, y_trainval)

    base_train_prob = base_model.predict_proba(X_trainval)[:, 1]
    base_test_prob = base_model.predict_proba(X_test)[:, 1]

    base_train_pred = base_model.predict(X_trainval)
    base_test_pred = base_model.predict(X_test)

    print("baseline train auc:", roc_auc_score(y_trainval, base_train_prob))
    print("baseline test auc:", roc_auc_score(y_test, base_test_prob))
    print("baseline test f1:", f1_score(y_test, base_test_pred))
    print("baseline test acc:", accuracy_score(y_test, base_test_pred))
    print(confusion_matrix(y_test, base_test_pred))

    return base_model

def count_features(base_model, X_trainval):
    # number of one-hot columns in the logistic model

    base_model.named_steps["prep"].fit(X_trainval)
    print("model inputs after preprocessing:", len(base_model.named_steps["prep"].get_feature_names_out()))

def PCA_plot(X_trainval, y_trainval, base_model):
    # 3d PCA view, draggable in the notebook
    # I only plot a sample so this cell does not become too slow

    rng = np.random.default_rng(1)
    plot_index = rng.choice(X_trainval.index, size=min(2000, len(X_trainval)), replace=False)

    X_plot = base_model.named_steps["prep"].transform(X_trainval.loc[plot_index])

    if hasattr(X_plot, "toarray"):
        X_plot = X_plot.toarray()

    pca = PCA(n_components=3, random_state=1)
    X_pca = pca.fit_transform(X_plot)

    pca_df = pd.DataFrame({
        "pc1": X_pca[:, 0],
        "pc2": X_pca[:, 1],
        "pc3": X_pca[:, 2],
        "target": y_trainval.loc[plot_index].astype(str).values,
    })

    fig = px.scatter_3d(
        pca_df,
        x="pc1",
        y="pc2",
        z="pc3",
        color="target",
        opacity=0.5,
    )

    fig.show()
    print("explained variance:", pca.explained_variance_ratio_)

    os.makedirs("models", exist_ok=True)
    joblib.dump(base_model, "models/logistic_regression.pkl")

def main():
    X, y, num_cols, cat_cols = load_and_prepare_data()
    X_train, X_val, X_trainval, X_test, y_train, y_val, y_trainval, y_test = split(X, y)
    prep = logistic_regression_preprocessing(num_cols, cat_cols)
    best_C, base_res = tune_logistic_regression(X_train, X_val, y_train, y_val, prep)
    base_model = train_logistic_regression(X_trainval, y_trainval, X_test, y_test, prep, best_C)
    count_features(base_model, X_trainval)
    PCA_plot(X_trainval, y_trainval, base_model)

if __name__ == "__main__":
    main()