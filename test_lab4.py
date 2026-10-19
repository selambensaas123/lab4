
import numpy as np

from ensemble_models import run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput


def test_ensemble_model():
    # Create a small artificial dataset
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    data = EnsembleInput(
        train_features=X[:80],
        train_target=y[:80],
        test_features=X[80:],
        test_target=y[80:],
        random_seed=42,
    )

    result = run_ensemble_experiment(data)

    assert len(result.probabilities) == 20
    assert len(result.predictions) == 20
    assert np.all(result.probabilities >= 0)
    assert np.all(result.probabilities <= 1)
    assert set(np.unique(result.predictions)).issubset({0, 1})

    assert 0 <= result.roc_auc <= 1
    assert 0 <= result.accuracy <= 1
    assert 0 <= result.precision <= 1
    assert 0 <= result.recall <= 1
    assert 0 <= result.f1 <= 1
