import numpy as np
from module import *
import pytest


Mat, Mask = setup_data_random_dropout(N=8192, dtype=np.float32)


@pytest.mark.parametrize(
    "matrix, dropout",
    [
        (Mat, Mask),
    ],
)
def test_equivalent_output(matrix, dropout):
    out_npsum = np_prod_sum(matrix, dropout)
    out_npdot = np_einsum(matrix, dropout)
    np.testing.assert_array_equal(out_npsum, out_npdot)


@pytest.mark.benchmark(group="Random Dropout")
@pytest.mark.parametrize(
    "matrix, dropout",
    [(Mat, Mask)],
)
def test_np_prod_sum(benchmark, matrix, dropout):
    benchmark(np_prod_sum, matrix, dropout)


@pytest.mark.benchmark(group="Random Dropout")
@pytest.mark.parametrize(
    "matrix, dropout",
    [(Mat, Mask)],
)
def test_np_einsum(benchmark, matrix, dropout):
    benchmark(np_einsum, matrix, dropout)
