"""Tests for AutoTS custom functionality."""

import pandas as pd
import pytest

from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("autots", severity="none"),
    reason="autots not available",
)
def test_autots_prediction_intervals():
    """Test that AutoTS can predict intervals."""
    from sktime.datasets import load_airline
    from sktime.forecasting.autots import AutoTS

    y = load_airline()

    # Configure with specific interval
    coverage = 0.9
    forecaster = AutoTS(
        model_list="superfast",
        max_generations=1,
        num_validations=0,
        prediction_interval=coverage,
        random_seed=42,  # Ensure reproducibility
    )

    forecaster.fit(y, fh=[1, 2, 3])

    # Test successful prediction
    intervals = forecaster.predict_interval(coverage=coverage)

    assert isinstance(intervals, pd.DataFrame)
    assert intervals.shape == (3, 2)
    assert intervals.columns.nlevels == 3

    # Check values
    lower = intervals.iloc[:, 0]
    upper = intervals.iloc[:, 1]
    assert (upper >= lower).all()

    # Test failure with wrong coverage
    with pytest.raises(ValueError, match="AutoTS configured with prediction_interval"):
        forecaster.predict_interval(coverage=0.5)


@pytest.mark.skipif(
    not _check_soft_dependencies("autots", severity="none"),
    reason="autots not available",
)
def test_autots_tags():
    """Test that AutoTS has correct tags."""
    from sktime.forecasting.autots import AutoTS

    forecaster = AutoTS()
    assert forecaster.get_tag("capability:pred_int") is True
