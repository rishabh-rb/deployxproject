import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

def evaluate_model(name, clf, X_tr, y_tr, X_te, y_te):
    clf.fit(X_tr, y_tr)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_te)[:, 1]
    else:
        proba = clf.predict(X_te)

    thresholds = np.linspace(0.1, 0.9, 81)
    macro_f1_scores = [
        f1_score(y_te, (proba >= t).astype(int), average="macro")
        for t in thresholds
    ]
    best_t = thresholds[int(np.argmax(macro_f1_scores))]
    preds = (proba >= best_t).astype(int)

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_te, preds)),
        "precision": float(precision_score(y_te, preds, zero_division=0)),
        "recall": float(recall_score(y_te, preds, zero_division=0)),
        "f1": float(f1_score(y_te, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, proba)) if hasattr(clf, "predict_proba") else None,
        "best_threshold": float(best_t),
    }


def train_and_save(
    data_path="student_data_cleaned.csv",
    model_path="student_model.pkl",
    metrics_path="model_metrics.json",
):
    df = pd.read_csv(data_path)

    X = df.drop("Pass", axis=1)
    y = df["Pass"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    model_results = [
        evaluate_model(model_name, model, X_train, y_train, X_test, y_test)
        for model_name, model in models.items()
    ]

    params = {
        "n_estimators": [150, 300],
        "max_depth": [None, 8, 16],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": ["balanced"],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        params,
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_proba = best_model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 81)
    macro_f1_scores = [
        f1_score(y_test, (y_proba >= t).astype(int), average="macro")
        for t in thresholds
    ]
    best_threshold = thresholds[int(np.argmax(macro_f1_scores))]

    final_preds = (y_proba >= best_threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, final_preds)),
        "precision": float(precision_score(y_test, final_preds)),
        "recall": float(recall_score(y_test, final_preds)),
        "f1": float(f1_score(y_test, final_preds)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "best_threshold": float(best_threshold),
        "confusion_matrix": confusion_matrix(y_test, final_preds).tolist(),
        "classification_report": classification_report(y_test, final_preds, output_dict=True),
        "feature_names": list(X.columns),
        "thresholds": {
            "study_hours_p10": float(np.percentile(X["TotalStudyHours"], 10)),
            "attendance_p10": float(np.percentile(X["TotalAttendance"], 10)),
            "previous_grade_p10": float(np.percentile(X["PreviousGrade"], 10)),
        },
        "model_results": model_results,
    }

    print("Final Model Metrics:")
    print(json.dumps(metrics, indent=2))

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    train_and_save()
