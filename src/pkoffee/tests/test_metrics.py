import numpy as np
import pytest


def test_check_size_match():
    from pkoffee.metrics import check_size_match, SizeMismatchError

    check_size_match(np.array([1, 2, 3]), np.array([4, 5, 6]))

    with pytest.raises(SizeMismatchError):
        check_size_match(np.array([1, 2, 3]), np.array([4, 5]))


def test_compute_mae():
    from pkoffee.metrics import compute_mae, SizeMismatchError

    assert compute_mae(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0
    assert compute_mae(np.array([1, 2, 3]), np.array([2, 4, 6])) == 2.0
    with pytest.raises(SizeMismatchError):
        compute_mae([1, 2], [1, 2, 3])


# def test_compute_rmse():
#     from pkoffee.metrics import compute_rmse

#     assert compute_rmse(np.array([1, 2, 3]), np.array([4, 5, 6])) == 2


# def test_compute_r2():
#     from pkoffee.metrics import compute_r2

#     assert compute_r2(np.array([1, 2, 3]), np.array([4, 5, 6])) == 2
