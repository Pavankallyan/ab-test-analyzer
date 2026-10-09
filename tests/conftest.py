"""Shared fixtures for the ab-test-analyzer test suite."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    """A reproducibly seeded random generator for synthetic experiment data."""
    return np.random.default_rng(42)


@pytest.fixture
def conversions():
    """Binary conversion outcomes: test converts at 12%, control at 10%."""
    rng = np.random.default_rng(7)
    test = (rng.random(2_000) < 0.12).astype(float)
    control = (rng.random(2_000) < 0.10).astype(float)
    return test, control


@pytest.fixture
def revenue(rng):
    """Continuous revenue: test averages $5.20, control $4.60 (sd 3.0)."""
    test = rng.normal(5.20, 3.0, size=800)
    control = rng.normal(4.60, 3.0, size=800)
    return test, control


@pytest.fixture
def skewed(rng):
    """Skewed latency-like data where the nonparametric test shines."""
    test = rng.lognormal(mean=2.4, sigma=0.6, size=600)
    control = rng.lognormal(mean=2.6, sigma=0.6, size=600)
    return test, control
