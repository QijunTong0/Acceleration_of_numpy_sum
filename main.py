import numpy as np


def main():
    print("NumPy Configuration:")
    np.show_config()
    print("\nStarting Benchmark...")

    # 記事の構成に合わせて int32 と float32 で検証
    benchmark_suite("int32")
    benchmark_suite("float32")


if __name__ == "__main__":
    main()
