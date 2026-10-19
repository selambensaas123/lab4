
"""Run and compare ensemble models for Lab 4."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ensemble_models import run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput


ROOT = Path(__file__).resolve().parent
SEED = 42


def calculate_metrics(model, X, y):
    # Get positive-class probabilities
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_index]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
    }


# Load German Credit dataset
df = pd.read_csv(ROOT / "german.csv", sep=";")

X = df.drop(columns=["Creditability"]).to_numpy(dtype=float)
y = df["Creditability"].to_numpy(dtype=int)

# First split: training and final test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Second split: training and validation data
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=SEED,
    stratify=y_train,
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=SEED,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=SEED,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=SEED,
    ),
}

validation_results = {}

print("VALIDATION RESULTS")

for name, model in models.items():
    # Fit on training data, evaluate on validation data
    model.fit(X_fit, y_fit)
    metrics = calculate_metrics(model, X_val, y_val)
    validation_results[name] = metrics

    print(name, metrics)

# Select the best model using validation ROC-AUC
best_name = max(
    validation_results,
    key=lambda name: validation_results[name]["roc_auc"],
)

print("\nBest validation model:", best_name)

# Refit the selected model on the full training split
best_model = models[best_name]
best_model.fit(X_train, y_train)
test_metrics = calculate_metrics(best_model, X_test, y_test)

# Also run the required contract function
contract_data = EnsembleInput(
    train_features=X_train,
    train_target=y_train,
    test_features=X_test,
    test_target=y_test,
    random_seed=SEED,
)

contract_result = run_ensemble_experiment(contract_data)

contract_metrics = {
    "roc_auc": contract_result.roc_auc,
    "accuracy": contract_result.accuracy,
    "precision": contract_result.precision,
    "recall": contract_result.recall,
    "f1": contract_result.f1,
}

results = {
    "random_seed": SEED,
    "dataset": "german.csv",
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "validation_results": validation_results,
    "best_validation_model": best_name,
    "best_model_test_metrics": test_metrics,
    "submitted_model": contract_result.model_name,
    "submitted_model_test_metrics": contract_metrics,
}

# Save the actual measured numbers
(ROOT / "experiment_results.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nFINAL TEST RESULTS")
print("Best validation model:", test_metrics)
print("Submitted model:", contract_metrics)
print("\nSaved experiment_results.json")

# Generate a report using the real measured results
metric_names = ["roc_auc", "accuracy", "precision", "recall", "f1"]

table_lines = [
    "| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |",
    "|---|---:|---:|---:|---:|---:|",
]

for name, metrics in validation_results.items():
    values = " | ".join(f"{metrics[k]:.4f}" for k in metric_names)
    table_lines.append(f"| {name} | {values} |")

validation_table = "\n".join(table_lines)

report = f"""# Лабораторная работа №4

## Данные и подготовка

Использован набор данных `german.csv`.
Целевая переменная: `Creditability`.
Признаки: все остальные 20 столбцов.
Данные считаны с разделителем `;`.
Признаки преобразованы в числовой формат NumPy.

Разбиение: 80% train и 20% test, stratify по целевой переменной.
Внутри train дополнительно выделено 25% для validation.
Размер train: {len(X_train)}.
Размер test: {len(X_test)}.
Случайное зерно: {SEED}.

## Гипотезы и эксперименты

Проверены три ансамблевых метода:

1. Random Forest: 300 деревьев, max_depth=8, min_samples_leaf=3.
2. Extra Trees: 300 деревьев, min_samples_leaf=3.
3. Gradient Boosting: 150 деревьев, learning_rate=0.05, max_depth=2.

Гипотеза: ансамбли деревьев могут лучше учитывать нелинейные
зависимости между признаками, чем простая линейная модель.

Выбор модели выполнен по ROC-AUC на validation.
Test не использовался для подбора параметров.

## Метрики

Результаты на validation:

{validation_table}

Выбранная по validation модель: **{best_name}**.

Итоговые метрики выбранной модели на test:

- ROC-AUC: {test_metrics["roc_auc"]:.4f}
- Accuracy: {test_metrics["accuracy"]:.4f}
- Precision: {test_metrics["precision"]:.4f}
- Recall: {test_metrics["recall"]:.4f}
- F1: {test_metrics["f1"]:.4f}

Метрики модели, возвращаемой контрактной функцией:

- Model: {contract_result.model_name}
- ROC-AUC: {contract_result.roc_auc:.4f}
- Accuracy: {contract_result.accuracy:.4f}
- Precision: {contract_result.precision:.4f}
- Recall: {contract_result.recall:.4f}
- F1: {contract_result.f1:.4f}

## Выбранная модель

Для сравнения выбрана **{best_name}**, так как она показала
максимальный ROC-AUC на validation.

Контрактная функция `run_ensemble_experiment` реализует Random Forest.
Это воспроизводимая модель с фиксированными параметрами.

## Выводы и ограничения

Ансамбли позволяют учитывать нелинейные зависимости.
Результаты зависят от разбиения и параметров модели.
Набор данных небольшой, поэтому оценки могут изменяться
при другом random seed.

Для более надёжного сравнения в будущем можно применить
кросс-валидацию и расширить поиск гиперпараметров.

## Воспроизводимость

Python, NumPy, Pandas, scikit-learn.
Random seed: {SEED}.
Разбиение: 80/20, stratified.
Для повторения:

```bash
python run_experiments.py
```

Численные результаты также сохранены в `experiment_results.json`.
"""

(ROOT / "REPORT.md").write_text(report, encoding="utf-8")
print("Saved REPORT.md")
