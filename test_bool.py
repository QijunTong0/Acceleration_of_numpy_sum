import numpy as np
from module import *
import pytest


Mat, Mask = setup_data_col_dropout(N=8192, dtype=np.bool_)


@pytest.mark.parametrize(
    "matrix, dropout",
    [
        (Mat, None),
        (Mat, Mask),
    ],
    ids=["all", "dropout"],
)
def test_equivalent_output(matrix, dropout):
    out_npsum = np_sum(matrix, dropout)
    out_npbitcount = np_bitcount(matrix, dropout)
    np.testing.assert_array_equal(out_npsum, out_npbitcount)


@pytest.mark.benchmark(group="BOOL")
@pytest.mark.parametrize(
    "matrix, dropout",
    [
        (Mat, None),
        (Mat, Mask),
    ],
    ids=["all", "dropout"],
)
def test_npsum(benchmark, matrix, dropout):
    benchmark(np_sum, matrix, dropout)


@pytest.mark.benchmark(group="BOOL")
@pytest.mark.parametrize(
    "matrix, dropout",
    [
        (Mat, None),
        (Mat, Mask),
    ],
    ids=["all", "dropout"],
)
def test_np_bitcount(benchmark, matrix, dropout):
    benchmark(np_bitcount, matrix, dropout)
