import numpy as np
from module import *
import pytest


Mat, Mask = setup_data_col_dropout(N=8192, dtype=np.float32)


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
    out_npsum_indexing = np_sum_indexing(matrix, dropout)
    out_npdot = np_dot(matrix, dropout)
    np.testing.assert_array_equal(out_npsum, out_npdot)
    np.testing.assert_array_equal(out_npsum, out_npsum_indexing)


@pytest.mark.benchmark(group="FLOAT32")
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


@pytest.mark.benchmark(group="FLOAT32")
@pytest.mark.parametrize(
    "matrix, dropout",
    [
        (Mat, Mask),
    ],
    ids=["dropout"],
)
def test_npsum_indexing(benchmark, matrix, dropout):
    benchmark(np_sum_indexing, matrix, dropout)


@pytest.mark.benchmark(group="FLOAT32")
@pytest.mark.parametrize(
    "matrix, dropout",
    [
        (Mat, None),
        (Mat, Mask),
    ],
    ids=["all", "dropout"],
)
def test_npdot(benchmark, matrix, dropout):
    benchmark(np_dot, matrix, dropout)
