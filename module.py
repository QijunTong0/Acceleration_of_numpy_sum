import numpy as np
import time


def setup_data(N=5000, dtype):
    """
    検証用のデータを作成します。
    A: NxNのランダム行列
    b: 全てが1のベクトル (単純和の代用)
    cond: 0か1のランダムベクトル (条件付き和のマスク)
    cond_bool: ブール型の条件ベクトル
    """
    print(f"Generating data (N={N}, dtype={dtype.__name__})...")

    # 0~100の範囲でランダムな整数を生成し、指定されたdtypeに変換
    # 記事に合わせて randint を使用しつつ dtype をキャスト
    A = np.random.randint(0, 100, size=(N, N)).astype(dtype)

    # 単純和用のベクトル (全て1)
    b = np.ones(N, dtype=dtype)

    # 条件付き和用のベクトル (0 or 1)
    # np.dot用には数値型(0, 1)が必要
    dropout = np.random.randint(0, 2, size=N).astype(dtype)

    # np.sum やフィルタリング用にはbool型やそのままの型を使用
    # (A * cond) の計算用に形状を合わせる（ブロードキャスト用）
    # condは1次元なのでそのままで行ごとの演算にブロードキャスト可能

    return A, b, dropout


def measure_execution_time(func, iterations=10, warmup=2):
    """
    関数の実行時間を計測します。
    """
    # ウォームアップ（キャッシュやJITの影響を排除）
    for _ in range(warmup):
        func()

    start_time = time.perf_counter()
    for _ in range(iterations):
        func()
    end_time = time.perf_counter()

    avg_time_ms = ((end_time - start_time) / iterations) * 1000
    return avg_time_ms


def benchmark_suite(dtype_str):
    """
    指定された型(int32, float32)でベンチマークを実行します。
    """
    dtype = getattr(np, dtype_str)
    N = 5000  # 記事の条件に合わせる
    ITERATIONS = 5  # 重い処理なので回数は少なめに設定

    A, b, cond = setup_data(N, dtype)

    print(f"\n=== Benchmark Result for {dtype.__name__} ===")
    print(f"{'Method':<35} | {'Time (ms)':<10} | {'Speedup'}")
    print("-" * 60)

    # ---------------------------------------------------------
    # 1. 条件なしの和 (Unconditional Sum)
    # ---------------------------------------------------------

    # Method A: A.sum(axis=1)
    def task_sum_naive():
        return A.sum(axis=1)

    # Method B: np.dot(A, b)
    def task_dot_product():
        return np.dot(A, b)

    # 正当性検証
    res_sum = task_sum_naive()
    res_dot = task_dot_product()
    # 浮動小数点の誤差を考慮して近似比較
    assert np.allclose(res_sum, res_dot, rtol=1e-5, atol=1e-5), (
        "Mismatch in Unconditional Sum!"
    )

    t_sum = measure_execution_time(task_sum_naive, iterations=ITERATIONS)
    t_dot = measure_execution_time(task_dot_product, iterations=ITERATIONS)

    print(f"{'A.sum(axis=1)':<35} | {t_sum:10.2f} | 1.00x (Baseline)")
    print(f"{'np.dot(A, b)':<35} | {t_dot:10.2f} | {t_sum / t_dot:.2f}x")

    # ---------------------------------------------------------
    # 2. 条件付きの和 (Conditional Sum)
    # ---------------------------------------------------------

    # Method A: (A * cond).sum(axis=1)
    # 記事にある実装。一時配列 (A*cond) が生成されるため遅いとされる。
    def task_conditional_sum_naive():
        return (A * cond).sum(axis=1)

    # Method B: A.sum(axis=1, where=cond_bool)
    # 記事でも言及されている where 引数を使うパターン (参考用)
    cond_bool = cond.astype(bool)

    def task_conditional_sum_where():
        return A.sum(axis=1, where=cond_bool)

    # Method C: np.dot(A, cond)
    # 記事で推奨されている高速化手法
    def task_conditional_dot():
        return np.dot(A, cond)

    # 正当性検証
    res_cond_sum = task_conditional_sum_naive()
    res_cond_dot = task_conditional_dot()
    assert np.allclose(res_cond_sum, res_cond_dot, rtol=1e-5, atol=1e-5), (
        "Mismatch in Conditional Sum!"
    )

    t_c_sum = measure_execution_time(task_conditional_sum_naive, iterations=ITERATIONS)
    # t_c_where = measure_execution_time(task_conditional_sum_where, iterations=ITERATIONS) # 参考
    t_c_dot = measure_execution_time(task_conditional_dot, iterations=ITERATIONS)

    print("-" * 60)
    print(f"{'(A*cond).sum(axis=1)':<35} | {t_c_sum:10.2f} | 1.00x (Baseline)")
    print(f"{'np.dot(A, cond)':<35} | {t_c_dot:10.2f} | {t_c_sum / t_c_dot:.2f}x")
    print("-" * 60)
