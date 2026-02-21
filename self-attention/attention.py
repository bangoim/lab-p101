import numpy as np


def softmax(x):
    max_per_row = np.max(x, axis=1, keepdims=True)
    shifted_values = x - max_per_row
    exponentials = np.exp(shifted_values)
    row_sums = np.sum(exponentials, axis=1, keepdims=True)
    return exponentials / row_sums


def scaled_dot_product_attention(Q, K, V):
    dk = K.shape[-1]
    scaling_factor = np.sqrt(dk)

    scores = Q @ K.T
    scaled_scores = scores / scaling_factor

    attention_weights = softmax(scaled_scores)
    output = attention_weights @ V

    return output, attention_weights
