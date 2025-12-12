"""
doubt - Discover hidden assumptions about data completeness

A tool to help developers see how missing data affects their functions.
"""

from .doubt import (
    DoubtResult,
    ImpactType,
    Scenario,
    assert_missing_robust,
    doubt,
)

__version__ = "0.1.0"
__all__ = [
    "doubt",
    "assert_missing_robust",
    "DoubtResult",
    "Scenario",
    "ImpactType",
]
