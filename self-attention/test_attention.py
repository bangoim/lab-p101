import numpy as np
from numpy.testing import assert_array_almost_equal
from attention import scaled_dot_product_attention


Q = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
], dtype=np.float64)

K = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
], dtype=np.float64)

V = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
], dtype=np.float64)


def compute_expected_output():
    dk = K.shape[-1]
    scaling_factor = np.sqrt(dk)
    scores = Q @ K.T
    scaled_scores = scores / scaling_factor
    max_per_row = np.max(scaled_scores, axis=1, keepdims=True)
    shifted = scaled_scores - max_per_row
    exponentials = np.exp(shifted)
    row_sums = np.sum(exponentials, axis=1, keepdims=True)
    expected_weights = exponentials / row_sums
    expected_output = expected_weights @ V
    return expected_output, expected_weights


def test_weights_sum_to_one(attention_weights):
    row_sums = np.sum(attention_weights, axis=1)
    expected_sums = np.ones(attention_weights.shape[0])
    try:
        assert_array_almost_equal(row_sums, expected_sums, decimal=6)
        print("  test_weights_sum_to_one: PASSED")
        return True
    except AssertionError:
        print("  test_weights_sum_to_one: FAILED")
        return False


def test_output_shape(output):
    expected_shape = (Q.shape[0], V.shape[1])
    if output.shape == expected_shape:
        print(f"  test_output_shape: PASSED (shape={output.shape})")
        return True
    else:
        print(f"  test_output_shape: FAILED (esperado={expected_shape}, obtido={output.shape})")
        return False


def test_numerical_correctness(output, attention_weights):
    expected_output, expected_weights = compute_expected_output()
    try:
        assert_array_almost_equal(attention_weights, expected_weights, decimal=6)
        assert_array_almost_equal(output, expected_output, decimal=6)
        print("  test_numerical_correctness: PASSED")
        return True
    except AssertionError:
        print("  test_numerical_correctness: FAILED")
        return False


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
    print(f"\nAttention Weights (shape={attention_weights.shape}):\n{attention_weights}")
    print(f"\nOutput (shape={output.shape}):\n{output}")

    print("\n" + "=" * 50)
    print("TESTES")
    print("=" * 50)

    results = [
        test_weights_sum_to_one(attention_weights),
        test_output_shape(output),
        test_numerical_correctness(output, attention_weights),
    ]

    total = len(results)
    passed = sum(results)
    print(f"\nResultado: {passed}/{total} testes passaram.")
