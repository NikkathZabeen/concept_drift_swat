from src.evaluation.metrics import classification_metrics, prequential_curve


def test_classification_metrics_binary():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]
    m = classification_metrics(y_true, y_pred)
    assert 0 <= m["f1"] <= 1
    assert m["accuracy"] == 0.5


def test_prequential_curve():
    curve = prequential_curve([1, 0, 1, 1], window=2)
    assert len(curve) == 4
    assert curve[-1] == 1.0
