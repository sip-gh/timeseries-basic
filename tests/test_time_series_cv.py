# tests/test_time_series_cv.py

import numpy as np
from src.time_series_cv import make_yearly_expanding_splits


def test_make_yearly_expanding_splits_basic():
    # 2013〜2017の5年分を想定したおもちゃデータ
    years = np.arange(2013, 2018)  # [2013, 2014, 2015, 2016, 2017]

    splits = make_yearly_expanding_splits(years, n_folds=3)

    # 3fold 返ってくるはず
    assert len(splits) == 3

    # foldごとの train/test 年のチェック
    expected = [
        (np.array([2013, 2014]),       np.array([2015])),               # Fold1
        (np.array([2013, 2014, 2015]), np.array([2016])),               # Fold2
        (np.array([2013, 2014, 2015, 2016]), np.array([2017])),         # Fold3
    ]

    for (train_mask, test_mask), (exp_train_years, exp_test_years) in zip(splits, expected):
        train_years = years[train_mask]
        test_years = years[test_mask]

        # 中身と順序が一致しているかを確認
        assert np.array_equal(train_years, exp_train_years)
        assert np.array_equal(test_years, exp_test_years)
