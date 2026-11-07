"""Final test labels cannot influence fitted models or calibrated intervals."""

import numpy as np

from forecast_exe.data import synthetic
from forecast_exe.model import calibrate, features, partitions, train


def test_final_test_isolation():
    df, meta = synthetic()
    _, _, test = partitions(df.index[29:-7])
    original = train(df, meta)
    changed = df.copy()
    changed.loc[test.min() :, ["high", "low"]] += 8
    changed.loc[test.min() :, "rain"] = 0
    revised = train(changed, meta)
    probe = features(df, 1).iloc[[100]]
    for h in range(1, 8):
        for target in ["high", "low", "rain"]:
            np.testing.assert_array_equal(
                original["models"][h, target].predict(probe), revised["models"][h, target].predict(probe)
            )
        for target in ["high", "low"]:
            np.testing.assert_array_equal(original["intervals"][h, target], revised["intervals"][h, target])

        np.testing.assert_array_equal(
            calibrate(original["calibrators"][h], [0.1, 0.5, 0.9]),
            calibrate(revised["calibrators"][h], [0.1, 0.5, 0.9]),
        )
