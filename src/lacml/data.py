"""Synthetic pediatric vital-signs data with repeated measurements per patient.

No real patient data is included. The generator deliberately gives each patient a
stable "signature" (baseline offsets) so that row-level splits leak patient identity
into the test fold -- the classic failure mode this project demonstrates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["heart_rate", "resp_rate", "spo2", "temperature", "systolic_bp"]


def make_vitals(n_patients: int = 120, records_per_patient: int = 15,
                signal: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Return one row per vital-signs record with columns FEATURES + label + patient_id.

    Label is assigned per patient (~30% positive, imbalanced). `signal` controls how
    much the condition truly shifts vitals; patient baselines are much stronger, so a
    model can 'cheat' by memorising patients when splits are not grouped.
    """
    rng = np.random.default_rng(seed)
    base = np.array([110.0, 30.0, 97.0, 37.0, 95.0])
    scale = np.array([12.0, 5.0, 1.5, 0.5, 8.0])

    labels = (rng.random(n_patients) < 0.30).astype(int)
    patient_offset = rng.normal(0.0, 1.0, size=(n_patients, len(FEATURES)))
    effect = np.array([1.0, 1.0, -1.0, 1.0, -0.5]) * signal

    rows = []
    for pid in range(n_patients):
        mu = base + scale * (patient_offset[pid] + labels[pid] * effect)
        noise = rng.normal(0.0, 0.25, size=(records_per_patient, len(FEATURES))) * scale
        for rec in mu + noise:
            rows.append([*rec, labels[pid], pid])
    return pd.DataFrame(rows, columns=[*FEATURES, "label", "patient_id"])
