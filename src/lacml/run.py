"""Reproduce the leakage experiment: python -m lacml.run"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .data import make_vitals
from .evaluate import cross_validate, nested_cross_validate

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main(out_dir: str = "results") -> dict:
    set_seeds()
    df = make_vitals(seed=SEED)
    results = {
        "row_split_rf (LEAKY)": cross_validate(df, "row", "rf", seed=SEED),
        "patient_split_rf": cross_validate(df, "patient", "rf", seed=SEED),
        "patient_split_logreg": cross_validate(df, "patient", "logreg", seed=SEED),
        "nested_patient_rf": nested_cross_validate(df, seed=SEED),
    }
    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir, "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"{'setting':<26}{'AUROC':>8}{'AUPRC':>8}")
    for k, v in results.items():
        print(f"{k:<26}{v['auroc']:>8.3f}{v['auprc']:>8.3f}")
    return results


if __name__ == "__main__":
    main()
