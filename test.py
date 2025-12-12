"""
Realistic demo for doubt: revenue analytics with messy data.

Run:
    python examples/complex_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from doubt import doubt


@doubt(max_scenarios_per_arg=8, change_threshold=0.03)
def compute_weighted_revenue(
    df: pd.DataFrame,
    tax_rate: float,
    region_multiplier: float = 1.0,
) -> float:
    revenue = df["price"] * df["quantity"]

    # weighted average revenue
    weighted_rev = (revenue * df["weight"]).sum() / df["weight"].sum()

    # tax + region adjustment
    net = weighted_rev * (1 - tax_rate)
    return net * region_multiplier


def main():
    df = pd.DataFrame(
        {
            "price": [100, 200, 150, 300],
            "quantity": [1, 2, 1, 3],
            "weight": [0.2, 0.3, 0.1, 0.4],
        }
    )

    print("\n=== Baseline Data ===")
    print(df)

    print("\n=== Running doubt analysis ===")

    result = compute_weighted_revenue.check(
        df=df,
        tax_rate=0.18,
        region_multiplier=1.2,
    )

    result.show()

    print("\n=== Top Risks ===")
    for s in result.get_risks():
        print(
            f"- {s.arg_name}{s.location}: "
            f"{s.impact_type.value}, "
            f"delta={s.relative_delta}"
        )


if __name__ == "__main__":
    main()
