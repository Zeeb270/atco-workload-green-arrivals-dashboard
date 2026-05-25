import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


FEATURE_COLUMNS = [
    "n_aircraft",
    "avg_distance_km",
    "std_distance_km",
    "avg_altitude_ft",
    "std_altitude_ft",
    "avg_speed_kt",
    "std_speed_kt",
    "n_runways",
    "min_arrival_gap_min",
    "mean_arrival_gap_min",
    "n_tight_arrival_gaps",
]


def get_model(model_name):
    if model_name == "kNN":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=3)),
        ])

    if model_name == "SVC":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True)),
        ])

    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=42)

    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )


def train_and_evaluate_model(features_df, model_name="Random Forest"):
    df = features_df.copy()

    X = df[FEATURE_COLUMNS]
    y = df["workload_label"]

    model = get_model(model_name)

    # Demo mode: with small sample data, train and evaluate on the same data.
    model.fit(X, y)
    y_pred = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "macro_precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
        "evaluation_mode": "Demo mode: trained and evaluated on the same sample dataset",
    }

    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)

    confusion_df = pd.DataFrame(
        cm,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels],
    )

    predictions_df = df.copy()
    predictions_df["ml_predicted_workload"] = y_pred

    feature_importance_df = get_feature_importance(model, X)

    return model, metrics, confusion_df, predictions_df, feature_importance_df


def get_feature_importance(model, X):
    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    else:
        importance_values = X.var().values
        if importance_values.sum() > 0:
            importance_values = importance_values / importance_values.sum()

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importance_values
    }).sort_values("importance", ascending=False)

    return importance_df
