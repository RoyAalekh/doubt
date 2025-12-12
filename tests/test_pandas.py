import pytest

from doubt import doubt
from doubt.doubt import ImpactType

pd = pytest.importorskip("pandas")




def test_pandas_series_handling():
    @doubt()
    def f(s):
        return s.sum()

    result = f.check(pd.Series([1, 2, 3]))
    assert result.scenarios

def test_dataframe_single_cell_missing():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [10.0, 20.0, 30.0],
        }
    )

    @doubt()
    def f(data):
        return data["a"].sum()

    result = f.check(df)

    # Baseline sanity
    assert result.baseline_output == 6.0

    # At least one scenario should be a silent change
    impacts = [s.impact_type for s in result.scenarios]

    assert ImpactType.SILENT_CHANGE in impacts

    # Original DataFrame must be unchanged
    assert df.isna().sum().sum() == 0

def test_top_customer_realistic_pandas():
    df = pd.DataFrame(
        {
            "customer_id": ["A", "A", "B", "B"],
            "price": [100.0, 50.0, 200.0, 10.0],
            "quantity": [1, 2, 1, 5],
        }
    )

    @doubt()
    def top_customer(df):
        revenue = (
            df.assign(revenue=df["price"] * df["quantity"])
              .groupby("customer_id", as_index=False)["revenue"]
              .sum()
        )
        return revenue.sort_values("revenue", ascending=False).iloc[0]["customer_id"]

    result = top_customer.check(df)

    # Baseline sanity
    assert result.baseline_output == "B"

    # There should be at least one concerning scenario
    assert any(s.is_concerning for s in result.scenarios)

    # Missing price or quantity should cause either:
    # - a crash
    # - or a silent change in top customer
    impacts = {s.impact_type for s in result.scenarios}
    assert ImpactType.CRASH in impacts or ImpactType.SILENT_CHANGE in impacts

    # Original dataframe must remain unchanged
    assert df.isna().sum().sum() == 0
    