"""
doubt - Discover hidden assumptions about data completeness

A tool to help developers see how missing data affects their functions.

"""

from __future__ import annotations

import functools
import inspect
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False


class ImpactType(Enum):
    """Types of impact from missing data"""

    CRASH = "crash"
    SILENT_CHANGE = "silent_change"
    TYPE_CHANGE = "type_change"
    NO_CHANGE = "no_change"


@dataclass
class Scenario:
    """A single missing data scenario"""

    arg_name: str
    location: str
    marker_used: Any
    output: Any = None
    error: Optional[str] = None
    impact_type: ImpactType = ImpactType.NO_CHANGE
    delta: Optional[float] = None
    relative_delta: Optional[float] = None
    
    @property
    def is_crash(self) -> bool:
        return self.impact_type == ImpactType.CRASH

    @property
    def is_concerning(self) -> bool:
        """Whether this scenario should concern developers"""
        return self.impact_type in (
            ImpactType.CRASH,
            ImpactType.SILENT_CHANGE,
            ImpactType.TYPE_CHANGE,
        )

@dataclass
class DoubtResult:
    """Results from missing data analysis"""

    func_name: str
    baseline_output: Any = None
    baseline_error: Optional[str] = None
    scenarios: List[Scenario] = field(default_factory=list)

    @property
    def crashes(self) -> List[Scenario]:
        return [s for s in self.scenarios if s.impact_type == ImpactType.CRASH]

    @property
    def silent_changes(self) -> List[Scenario]:
        return [s for s in self.scenarios if s.impact_type == ImpactType.SILENT_CHANGE]

    @property
    def type_changes(self) -> List[Scenario]:
        return [s for s in self.scenarios if s.impact_type == ImpactType.TYPE_CHANGE]

    @property
    def baseline_ok(self) -> bool:
        return self.baseline_error is None

    def show(self, max_scenarios: int = 20) -> None:
        """Pretty print the analysis results"""
        print(f"\nDoubt Analysis: {self.func_name}()\n")
        print("=" * 70)

        if not self.baseline_ok:
            print(f"\nBASELINE CRASHED: {self.baseline_error}")
            print("\nFix your function first before analyzing missing data impact!")
            return

        print(f"\nBaseline Output: {_format_value(self.baseline_output)}")

        # Summary stats
        total = len(self.scenarios)
        crashes = len(self.crashes)
        silent = len(self.silent_changes)
        type_changes = len(self.type_changes)

        print(f"\nTested {total} scenarios:")
        print(f"   Crashes: {crashes}")
        print(f"   Silent Changes: {silent}")
        print(f"   Type Changes: {type_changes}")
        print(f"   No Impact: {total - crashes - silent - type_changes}")

        # Show concerning scenarios
        concerning = [s for s in self.scenarios if s.is_concerning]

        if not concerning:
            print("\nGreat! Your function handles missing data gracefully!")
            return

        print(f"\nConcerning Scenarios (showing up to {max_scenarios}):\n")
        print(f"{'Argument':<20} {'Location':<15} {'Impact':<15} {'Details':<30}")
        print("-" * 80)

        for scenario in concerning[:max_scenarios]:
            impact = _format_impact(scenario.impact_type)
            details = _format_details(scenario)
            print(f"{scenario.arg_name:<20} {scenario.location:<15} {impact:<15} {details:<30}")

        if len(concerning) > max_scenarios:
            print(f"\n... and {len(concerning) - max_scenarios} more scenarios")

        # Suggestions
        self._show_suggestions()

    def _show_suggestions(self) -> None:
        """Show actionable suggestions based on findings"""
        if not self.baseline_ok:
            return

        print("\nSuggestions:\n")

        if self.crashes:
            print("   - Add defensive checks for None/NaN values")
            print("   - Consider using try-except blocks")
            args_that_crash = set(s.arg_name for s in self.crashes)
            print(f"   - Problematic arguments: {', '.join(sorted(args_that_crash))}")

        if self.silent_changes:
            print("   - Document assumptions about data completeness")
            print("   - Add explicit handling for missing values")
            print("   - Consider raising errors instead of silently changing output")

        if self.type_changes:
            print("   - Type changes can cause downstream errors")
            print("   - Consider always returning the same type")

    def get_risks(self, top_n: int = 5) -> List[Scenario]:
        """Get top N riskiest scenarios"""
        # Priority: crashes > type changes > silent changes > no change
        priority = {
            ImpactType.CRASH: 3,
            ImpactType.TYPE_CHANGE: 2,
            ImpactType.SILENT_CHANGE: 1,
            ImpactType.NO_CHANGE: 0,
        }

        sorted_scenarios = sorted(
            self.scenarios,
            key=lambda s: (
                priority[s.impact_type],
                abs(s.relative_delta) if s.relative_delta else 0,
            ),
            reverse=True,
        )

        return sorted_scenarios[:top_n]


# Helper functions for formatting
def _format_value(value: Any, max_len: int = 50) -> str:
    """Format value for display"""
    if value is None:
        return "None"

    s = str(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _format_impact(impact: ImpactType) -> str:
    """Format impact type"""
    labels = {
        ImpactType.CRASH: "Crash",
        ImpactType.SILENT_CHANGE: "Changed",
        ImpactType.TYPE_CHANGE: "Type Change",
        ImpactType.NO_CHANGE: "No Impact",
    }
    return labels.get(impact, str(impact))


def _format_details(scenario: Scenario) -> str:
    """Format scenario details"""
    if scenario.error:
        # Extract just the error type
        error_type = scenario.error.split(":")[0] if ":" in scenario.error else scenario.error
        return f"{error_type}"

    if scenario.relative_delta is not None:
        return f"{scenario.relative_delta:+.1%}"

    if scenario.impact_type == ImpactType.TYPE_CHANGE:
        return "type changed"

    return "-"


# Core missing marker logic
def _choose_marker(value: Any, user_marker: Any) -> Any:
    """Choose appropriate missing marker for a value type"""
    if user_marker is not None:
        return user_marker

    # For numeric scalars, use NaN
    if isinstance(value, float):
        return float("nan")

    # For numpy arrays with float dtype, use NaN
    if HAS_NUMPY and isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating):
            return np.nan

    # For pandas, use NaN
    if HAS_PANDAS and isinstance(value, (pd.Series, pd.DataFrame)):
        return np.nan if HAS_NUMPY else None

    # Default to None
    return None


def _is_numeric(x: Any) -> bool:
    """Check if a value is numeric (and not NaN)"""
    if isinstance(x, bool):
        return False

    if isinstance(x, (int, float)):
        return not (isinstance(x, float) and math.isnan(x))

    if HAS_NUMPY and isinstance(x, np.number):
        return not np.isnan(x)

    return False


def _values_equal(a: Any, b: Any) -> bool:
    """Check if two values are equal, handling special cases"""
    try:
        # Handle numpy arrays
        if HAS_NUMPY and isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.array_equal(a, b, equal_nan=True)

        # Handle pandas
        if HAS_PANDAS:
            if isinstance(a, (pd.Series, pd.DataFrame)) and isinstance(
                b, (pd.Series, pd.DataFrame)
            ):
                return a.equals(b)

        # Handle NaN
        if _is_nan(a) and _is_nan(b):
            return True

        # Regular equality
        return a == b

    except (ValueError, TypeError):
        # If comparison fails, consider them different
        return False


def _is_nan(x: Any) -> bool:
    """Check if value is NaN"""
    if isinstance(x, float):
        return math.isnan(x)
    if HAS_NUMPY and isinstance(x, np.number):
        return np.isnan(x)
    return False


def _classify_impact(
    baseline: Any, output: Any, error: Optional[str], threshold: float = 0.05
) -> Tuple[ImpactType, Optional[float], Optional[float]]:
    """Classify the type of impact from missing data"""

    # Crash
    if error is not None:
        return ImpactType.CRASH, None, None

    # No change
    if _values_equal(baseline, output):
        return ImpactType.NO_CHANGE, 0.0, 0.0

    # Type change
    if type(baseline) is not type(output):
        return ImpactType.TYPE_CHANGE, None, None

    # Calculate numeric delta if both are numeric
    if _is_numeric(baseline) and _is_numeric(output):
        delta = float(output) - float(baseline)
        rel_delta = delta / float(baseline) if baseline != 0 else None

        # Check if change is below threshold
        if rel_delta is not None and abs(rel_delta) < threshold:
            return ImpactType.NO_CHANGE, delta, rel_delta

        return ImpactType.SILENT_CHANGE, delta, rel_delta

    # Non-numeric change
    return ImpactType.SILENT_CHANGE, None, None


# Perturbation generation
def _generate_perturbations(
    value: Any, marker: Any, max_per_arg: int = 10
) -> List[Tuple[Any, str]]:
    """Generate missing data perturbations for a value"""
    scenarios = []

    # Scalars
    if value is None or isinstance(value, (int, float, str, bool)):
        scenarios.append((marker, "scalar"))
        return scenarios

    # Lists
    if isinstance(value, list):
        n = min(len(value), max_per_arg)
        for i in range(n):
            new_list = value.copy()
            new_list[i] = marker
            scenarios.append((new_list, f"[{i}]"))
        return scenarios

    # Tuples
    if isinstance(value, tuple):
        n = min(len(value), max_per_arg)
        for i in range(n):
            new_list = list(value)
            new_list[i] = marker
            scenarios.append((tuple(new_list), f"[{i}]"))
        return scenarios

    # Dicts
    if isinstance(value, dict):
        keys = list(value.keys())[:max_per_arg]
        for k in keys:
            new_dict = value.copy()
            new_dict[k] = marker
            scenarios.append((new_dict, f"['{k}']"))
        return scenarios

    # NumPy arrays
    if HAS_NUMPY and isinstance(value, np.ndarray):
        flat = value.ravel()
        n = min(flat.size, max_per_arg)
        for i in range(n):
            new_arr = value.copy()
            new_flat = new_arr.ravel()
            new_flat[i] = marker
            scenarios.append((new_arr, f"[{i}]"))
        return scenarios

    # Pandas Series
    if HAS_PANDAS and isinstance(value, pd.Series):
        idxs = list(value.index)[:max_per_arg]
        for idx in idxs:
            new_series = value.copy()
            new_series.loc[idx] = marker
            scenarios.append((new_series, f"[{idx}]"))
        return scenarios

    # Pandas DataFrame
    if HAS_PANDAS and isinstance(value, pd.DataFrame):
        count = 0
        for row in value.index:
            for col in value.columns:
                if count >= max_per_arg:
                    break
                new_df = value.copy()
                new_df.loc[row, col] = marker
                scenarios.append((new_df, f"[{row}, '{col}']"))
                count += 1
            if count >= max_per_arg:
                break
        return scenarios

    # Unknown type - replace whole object
    scenarios.append((marker, "object"))
    return scenarios


def _has_complex_signature(sig: inspect.Signature) -> Tuple[bool, Optional[str]]:
    """Check if signature has features we don't fully support yet"""

    # Check for positional-only parameters
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            return True, "positional-only parameters (/) not fully supported"

    # Check for *args or **kwargs
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return True, "*args not fully supported"
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True, "**kwargs not fully supported"

    return False, None


# Main decorator
def doubt(
    *, missing_marker: Any = None, max_scenarios_per_arg: int = 10, change_threshold: float = 0.05
) -> Callable:
    """
    Decorator to analyze how missing data affects a function.

    Args:
        missing_marker: Value to use as "missing". None = auto-detect per type
        max_scenarios_per_arg: Max scenarios to test per argument
        change_threshold: Relative change threshold below which changes are ignored (default 5%)

    Example:
        @doubt()
        def calculate_mean(values):
            return sum(values) / len(values)

        result = calculate_mean.check([1, 2, 3, 4, 5])
        result.show()

    Limitations:
        - Functions with positional-only parameters (/) not fully supported
        - Functions with *args or **kwargs not fully supported
    """

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)

        # Check for unsupported signature features
        has_complex, reason = _has_complex_signature(sig)
        if has_complex:
            import warnings

            warnings.warn(
                f"doubt: {func.__name__} has {reason}. Analysis may not work correctly.",
                UserWarning,
            )

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            # Normal function call - no analysis
            return func(*args, **kwargs)

        def check(*args, **kwargs) -> DoubtResult:
            """Run missing data analysis on this function"""

            # Bind arguments using signature
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
            except TypeError as e:
                return DoubtResult(
                    func_name=func.__name__,
                    baseline_output=None,
                    baseline_error=f"Signature binding failed: {e}",
                    scenarios=[],
                )

            # Run baseline using bound args/kwargs (preserves positional semantics)
            try:
                baseline_output = func(*bound.args, **bound.kwargs)
                baseline_error = None
            except Exception as e:
                baseline_output = None
                baseline_error = f"{type(e).__name__}: {str(e)}"
                return DoubtResult(
                    func_name=func.__name__,
                    baseline_output=baseline_output,
                    baseline_error=baseline_error,
                    scenarios=[],
                )

            # Create result container
            result = DoubtResult(
                func_name=func.__name__,
                baseline_output=baseline_output,
                baseline_error=baseline_error,
            )

            # Test each argument with missing data
            for arg_name, value in bound.arguments.items():
                marker = _choose_marker(value, missing_marker)
                perturbations = _generate_perturbations(
                    value, marker, max_scenarios_per_arg
                )

                for perturbed_value, location in perturbations:
                    # Create new BoundArguments with perturbed value
                    try:
                        # Rebind from scratch to preserve semantics
                        new_bound = sig.bind(*bound.args, **bound.kwargs)
                        new_bound.arguments[arg_name] = perturbed_value

                        # Call with proper args/kwargs (handles positional-only correctly)
                        output = func(*new_bound.args, **new_bound.kwargs)
                        error = None
                    except Exception as e:
                        output = None
                        error = f"{type(e).__name__}: {str(e)}"

                    # Classify impact
                    impact_type, delta, rel_delta = _classify_impact(
                        baseline_output, output, error, change_threshold
                    )

                    # Create scenario
                    scenario = Scenario(
                        arg_name=arg_name,
                        location=location,
                        marker_used=marker,
                        output=output,
                        error=error,
                        impact_type=impact_type,
                        delta=delta,
                        relative_delta=rel_delta,
                    )

                    result.scenarios.append(scenario)

            return result

        # Attach check method to wrapped function
        wrapped.check = check
        return wrapped

    return decorator


# Pytest helper for robustness testing
def assert_missing_robust(
    func: Callable,
    *args,
    max_relative_change: float = 0.1,
    allow_type_changes: bool = False,
    **kwargs,
) -> None:
    """
    Assert that a function handles missing data robustly.

    Raises AssertionError if:
    - Function crashes with missing data
    - Relative output change exceeds max_relative_change
    - Output type changes (unless allow_type_changes=True)

    Example:
        def test_my_function_robustness():
            assert_missing_robust(
                my_function,
                [1, 2, 3, 4],
                max_relative_change=0.05
            )
    """
    if not hasattr(func, "check"):
        raise ValueError(
            f"{func.__name__} must be decorated with @doubt() to use assert_missing_robust"
        )

    result = func.check(*args, **kwargs)

    if not result.baseline_ok:
        raise AssertionError(f"Baseline crashed: {result.baseline_error}")

    errors = []

    # Check for crashes
    if result.crashes:
        crash_summary = ", ".join(f"{s.arg_name}{s.location}" for s in result.crashes[:3])
        if len(result.crashes) > 3:
            crash_summary += f" (+{len(result.crashes) - 3} more)"
        errors.append(f"Function crashes with missing data: {crash_summary}")

    # Check for excessive changes
    for scenario in result.silent_changes:
        if scenario.relative_delta is not None:
            if abs(scenario.relative_delta) > max_relative_change:
                errors.append(
                    f"{scenario.arg_name}{scenario.location}: "
                    f"change {scenario.relative_delta:+.1%} exceeds threshold {max_relative_change:.1%}"
                )

    # Check for type changes
    if not allow_type_changes and result.type_changes:
        type_change_summary = ", ".join(
            f"{s.arg_name}{s.location}" for s in result.type_changes[:3]
        )
        if len(result.type_changes) > 3:
            type_change_summary += f" (+{len(result.type_changes) - 3} more)"
        errors.append(f"Output type changes: {type_change_summary}")

    if errors:
        msg = "Missing data robustness check failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise AssertionError(msg)
