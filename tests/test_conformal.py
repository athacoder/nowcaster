import math

import numpy as np
import pandas as pd
import pytest

from nowcaster import conformal


def test_half_width_takes_the_conformal_rank() -> None:
    errors = np.arange(1.0, 10.0)  # nine errors: 1..9
    assert conformal.half_width(errors, 0.80) == 8.0  # ceil(10 * 0.8) = 8th smallest
    assert conformal.half_width(errors, 0.95) == math.inf  # ceil(10 * 0.95) = 10 > 9


def test_online_half_widths_never_use_the_current_or_later_months() -> None:
    index = pd.period_range("2016-01", periods=40, freq="M")
    errors = pd.Series(1.0, index=index)
    errors.iloc[38:] = 100.0  # two huge errors at the end
    widths = conformal.online_half_widths(errors, 0.80)
    assert widths.index[0] == index[36]
    assert widths[index[38]] == 1.0  # month 38 sees only months 0..37


def test_coverage_is_close_to_nominal_on_exchangeable_errors() -> None:
    rng = np.random.default_rng(0)
    index = pd.period_range("2000-01", periods=2000, freq="M")
    errors = pd.Series(np.abs(rng.standard_t(4, size=2000)), index=index)
    for level in conformal.LEVELS:
        share, _, months = conformal.coverage(errors, level)
        assert months == 2000 - conformal.MIN_HISTORY
        assert share == pytest.approx(level, abs=0.03)
