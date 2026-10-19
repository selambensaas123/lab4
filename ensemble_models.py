"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    # Convert input data to NumPy arrays
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target).ravel()
    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target).ravel()

    # Train only on the training data
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=data.random_seed,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Predict probabilities and class labels
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, positive_index]
    predictions = (probabilities >= 0.5).astype(int)

    # Calculate test metrics
    return EnsembleResult(
        model_name="Random Forest",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0)),
    )
