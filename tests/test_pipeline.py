import numpy as np

from lacml.data import make_vitals
from lacml.evaluate import cross_validate
from sklearn.model_selection import StratifiedGroupKFold


def test_reproducible_data():
    a, b = make_vitals(seed=1), make_vitals(seed=1)
    assert a.equals(b)


def test_no_patient_overlap_in_grouped_split():
    df = make_vitals(seed=0)
    g = df["patient_id"].to_numpy()
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, df["label"], g):
        assert set(g[tr]).isdisjoint(g[te])


def test_row_split_inflates_performance():
    df = make_vitals(seed=0)
    leaky = cross_validate(df, "row", "rf", seed=0)["auroc"]
    honest = cross_validate(df, "patient", "rf", seed=0)["auroc"]
    assert leaky > honest + 0.1


def test_metrics_are_deterministic():
    df = make_vitals(seed=0)
    r1 = cross_validate(df, "patient", "rf", seed=0)
    r2 = cross_validate(df, "patient", "rf", seed=0)
    assert np.isclose(r1["auroc"], r2["auroc"])
