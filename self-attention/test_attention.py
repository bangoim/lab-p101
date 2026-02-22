import numpy as np
from attention import scaled_dot_product_attention

Q: np.ndarray = np.array([
    [0.2, 0.8, 0.1],
    [0.9, 0.1, 0.5],
    [0.3, 0.6, 0.7],
    [0.5, 0.5, 0.0],
], dtype=np.float64)

K: np.ndarray = np.array([
    [0.6, 0.3, 0.4],
    [0.1, 0.9, 0.2],
    [0.7, 0.2, 0.8],
], dtype=np.float64)

V: np.ndarray = np.array([
    [1.0, 0.5, 0.0],
    [0.0, 1.0, 0.5],
    [0.5, 0.0, 1.0],
], dtype=np.float64)


def test_weights_sum_to_one(attention_weights: np.ndarray) -> bool:
    row_sums: np.ndarray = np.sum(attention_weights, axis=1)
    all_close: bool = np.allclose(row_sums, 1.0, atol=1e-6)
    if all_close:
        print("  test_weights_sum_to_one: PASSED")
    else:
        print(f"  test_weights_sum_to_one: FAILED (row sums: {row_sums})")
    return all_close


def test_weights_non_negative(attention_weights: np.ndarray) -> bool:
    all_non_negative: bool = bool(np.all(attention_weights >= 0.0))
    if all_non_negative:
        print("  test_weights_non_negative: PASSED")
    else:
        negative_count: int = int(np.sum(attention_weights < 0.0))
        print(f"  test_weights_non_negative: FAILED ({negative_count} negative values)")
    return all_non_negative


def test_output_shape(output: np.ndarray) -> bool:
    expected_shape = (Q.shape[0], V.shape[1])
    if output.shape == expected_shape:
        print(f"  test_output_shape: PASSED (shape={output.shape})")
        return True
    else:
        print(
            f"  test_output_shape: FAILED (expected={expected_shape}, obtained={output.shape})"
        )
        return False


def test_numerical_correctness(
    output: np.ndarray, attention_weights: np.ndarray
) -> bool:
    dk: int = K.shape[-1]
    raw_scores: np.ndarray = (Q @ K.T) / np.sqrt(dk)
    stabilized: np.ndarray = raw_scores - np.max(raw_scores, axis=1, keepdims=True)
    exp_scores: np.ndarray = np.exp(stabilized)
    expected_weights: np.ndarray = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    expected_output: np.ndarray = expected_weights @ V

    weights_match: bool = np.allclose(attention_weights, expected_weights, atol=1e-6)
    output_match: bool = np.allclose(output, expected_output, atol=1e-6)

    if weights_match and output_match:
        print("  test_numerical_correctness: PASSED")
    else:
        print("  test_numerical_correctness: FAILED")
        if not weights_match:
            print(f"    attention_weights mismatch (max diff: {np.max(np.abs(attention_weights - expected_weights)):.2e})")
        if not output_match:
            print(f"    output mismatch (max diff: {np.max(np.abs(output - expected_output)):.2e})")
    return weights_match and output_match


if __name__ == "__main__":
    print("=" * 50)
    print("INPUTS")
    print("=" * 50)
    print(f"\nQ (shape={Q.shape}):\n{Q}")
    print(f"\nK (shape={K.shape}):\n{K}")
    print(f"\nV (shape={V.shape}):\n{V}")

    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    print("\n" + "=" * 50)
    print("OUTPUTS")
    print("=" * 50)
    print(
        f"\nAttention Weights (shape={attention_weights.shape}):\n{attention_weights}"
    )
    print(f"\nOutput (shape={output.shape}):\n{output}")

    print("\n" + "=" * 50)
    print("TESTS")
    print("=" * 50)

    results = [
        test_weights_sum_to_one(attention_weights),
        test_weights_non_negative(attention_weights),
        test_output_shape(output),
        test_numerical_correctness(output, attention_weights),
    ]

    total = len(results)
    passed = sum(results)
    print(f"\nResult: {passed}/{total} tests passed.")
