from typing import Tuple

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(
            f"softmax expects a 2D matrix, but received an array with {x.ndim} dimension(s) instead."
        )

    max_per_row = np.max(x, axis=1, keepdims=True)
    shifted_values = x - max_per_row
    exponentials = np.exp(shifted_values)
    row_sums = np.sum(exponentials, axis=1, keepdims=True)
    return exponentials / row_sums


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    for name, matrix in [("Q", Q), ("K", K), ("V", V)]:
        if matrix.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D matrix, but has {matrix.ndim} dimension(s) instead."
            )

    if Q.shape[1] != K.shape[1]:
        raise ValueError(
            f"Incompatible dimension dₖ: Q.shape[1]={Q.shape[1]} != K.shape[1]={K.shape[1]}."
        )

    if K.shape[0] != V.shape[0]:
        raise ValueError(
            f"Incompatible number of rows: K.shape[0]={K.shape[0]} != V.shape[0]={V.shape[0]}."
        )

    dk = K.shape[-1]
    scaling_factor = np.sqrt(dk)

    scores = Q @ K.T
    scaled_scores = scores / scaling_factor

    attention_weights = softmax(scaled_scores)
    output = attention_weights @ V

    return output, attention_weights
