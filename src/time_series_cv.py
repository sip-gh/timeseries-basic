import numpy as np
import pandas as pd


def make_yearly_expanding_splits(
        years: np.ndarray | pd.Series,
        n_folds: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    years_arr = np.asarray(years)
    unique_years = np.sort(np.unique(years_arr))

    if len(unique_years) < n_folds + 1:
        raise ValueError(
            f"unique_years={len(unique_years)} だと"
            f"n_folds={n_folds} のexpanding CVは組めません"
        )

    cv_years = unique_years[-n_folds:]

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for test_year in cv_years:
        train_mask = years_arr < test_year
        test_mask = years_arr == test_year
        splits.append((train_mask, test_mask))

    return splits
