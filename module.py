import numpy as np


def setup_data_col_dropout(dtype, N=5000):
    """
    列単位での
    """
    A = np.random.randint(0, 100, size=(N, N)).astype(dtype)
    dropout = np.random.randint(0, 2, size=N).astype(dtype=np.bool_)
    return A, dropout


def setup_data_random_dropout(dtype, N=5000):
    A = np.random.randint(0, 100, size=(N, N)).astype(dtype)
    dropout = np.random.randint(0, 2, size=(N, N)).astype(dtype=np.bool_)
    return A, dropout


def np_sum(A: np.ndarray, dropout=None):
    if dropout is not None:
        return (A * dropout).sum(axis=1)
    else:
        return A.sum(axis=1)


def np_sum_indexing(A: np.ndarray, dropout=None):
    if dropout is not None:
        return A[:, dropout].sum(axis=1)
    else:
        return A.sum(axis=1)


def np_dot(A: np.ndarray, dropout=None):
    if dropout is not None:
        return np.dot(A, dropout)
    else:
        b = np.ones(len(A), dtype=A.dtype)
        return np.dot(A, b)


def np_bitcount(A: np.ndarray, dropout=None):
    if dropout is not None:
        A_packed = np.packbits(A, axis=1)
        A_view64 = A_packed.view(np.uint64)
        D_packed = np.packbits(dropout)
        D_view64 = D_packed.view(np.uint64)
        A_view64 &= D_view64
        counts = np.bitwise_count(A_view64)
        return counts.sum(axis=1)

    else:
        A_packed = np.packbits(A, axis=1)
        A_view64 = A_packed.view(np.uint64)
        counts = np.bitwise_count(A_view64)
        return counts.sum(axis=1)


def np_prod_sum(A: np.ndarray, dropout: np.ndarray):
    return (A * dropout).sum(axis=1)


def np_einsum(A: np.ndarray, dropout: np.ndarray):
    return np.einsum("ij,ij->i", A, dropout)
