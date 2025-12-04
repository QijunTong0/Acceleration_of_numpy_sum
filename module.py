import numpy as np


def setup_data(dtype, N=5000):
    """
    検証用のデータを作成します。
    A: NxNのランダム行列
    b: 全てが1のベクトル (単純和の代用)
    cond: 0か1のランダムベクトル (条件付き和のマスク)
    cond_bool: ブール型の条件ベクトル
    """

    # 0~100の範囲でランダムな整数を生成し、指定されたdtypeに変換
    # 記事に合わせて randint を使用しつつ dtype をキャスト
    A = np.random.randint(0, 100, size=(N, N)).astype(dtype)

    # 条件付き和用のベクトル (0 or 1)
    # np.dot用には数値型(0, 1)が必要
    dropout = np.random.randint(0, 2, size=N).astype(dtype=np.bool_)

    # np.sum やフィルタリング用にはbool型やそのままの型を使用
    # (A * cond) の計算用に形状を合わせる（ブロードキャスト用）
    # condは1次元なのでそのままで行ごとの演算にブロードキャスト可能

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
