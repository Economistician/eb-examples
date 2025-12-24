import pandas as pd
import numpy as np

import eb_metrics as m
import eb_evaluation as ev


def test_cwsl_array_and_df_match():
    y = np.array([10, 12, 9, 11], dtype=float)
    yhat = np.array([9, 12, 10, 8], dtype=float)

    cwsl_arr = m.cwsl(y, yhat, cu=2.0, co=1.0)

    df = pd.DataFrame({"y": y, "yhat": yhat})
    cwsl_df = ev.compute_cwsl_df(df, y_true_col="y", y_pred_col="yhat", cu=2.0, co=1.0)

    assert isinstance(cwsl_arr, float)
    assert isinstance(cwsl_df, float)
    assert cwsl_arr == cwsl_df


def test_evaluate_panel_df_smoke():
    df = pd.DataFrame(
        {
            "store": [101, 101, 202, 202],
            "date": ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"],
            "y": [10, 12, 9, 11],
            "yhat": [9, 12, 10, 8],
        }
    )

    levels = {"overall": [], "by_store": ["store"]}

    out = ev.evaluate_panel_df(
        df,
        levels=levels,
        actual_col="y",
        forecast_col="yhat",
        cu=2.0,
        co=1.0,
        tau=1.0,
    )

    # basic shape/contract checks
    assert set(["level", "metric", "value"]).issubset(out.columns)
    assert (out["level"] == "overall").any()
    assert (out["level"] == "by_store").any()

    # ensure key metrics are present
    metrics = set(out["metric"].unique())
    assert "cwsl" in metrics
    assert "frs" in metrics
    assert "wmape" in metrics
    assert "hr_at_tau" in metrics