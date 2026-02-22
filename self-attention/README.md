# Scaled Dot-Product Attention — LAB P1-01

## Descrição

Este repositório contém uma implementação do mecanismo de **Scaled Dot-Product Attention**, conforme descrito no paper *"Attention Is All You Need"* (Vaswani et al., 2017). O mecanismo de atenção é o componente central da arquitetura Transformer, permitindo que o modelo aprenda a ponderar a relevância de diferentes posições de uma sequência ao processar cada elemento.

O funcionamento se baseia em três matrizes — **Query (Q)**, **Key (K)** e **Value (V)** — derivadas da entrada. Para cada query, o mecanismo calcula a similaridade com todas as keys por meio de um produto escalar, normaliza os scores resultantes com a função softmax, e utiliza esses pesos para computar uma combinação ponderada dos values. O resultado é uma representação contextualizada onde cada posição incorpora informação das demais, proporcionalmente à relevância calculada.

A implementação utiliza apenas **NumPy** para toda a álgebra linear, sem dependências de frameworks de Deep Learning como PyTorch ou TensorFlow.

## Como Rodar

### Pré-requisitos

- Python 3.x
- NumPy

### Instalação e Execução

```bash
pip install -r requirements.txt
```

```bash
python test_attention.py
```

## Fórmula de Referência

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Onde:
- **Q** (Query): matriz de consultas, shape *(n, d_k)*
- **K** (Key): matriz de chaves, shape *(m, d_k)*
- **V** (Value): matriz de valores, shape *(m, d_v)*
- **d_k**: dimensão das chaves (última dimensão de K)

## Explicação do Scaling Factor (√d_k)

O produto escalar entre vetores de query e key produz valores cuja magnitude cresce proporcionalmente à dimensão *d_k*. Quando dois vetores de dimensão *d_k* têm componentes independentes com média 0 e variância 1, o valor esperado do produto escalar é 0, mas a **variância é d_k**. Isso significa que quanto maior a dimensão, maiores em magnitude os scores tendem a ser.

Scores de grande magnitude empurram a função softmax para regiões onde os gradientes são extremamente pequenos — a chamada **saturação**. Nessas regiões, o softmax produz distribuições quase one-hot, dificultando o aprendizado, pois os gradientes praticamente desaparecem.

Dividir os scores por **√d_k** normaliza a variância de volta para 1, independente da dimensão. Isso mantém o softmax operando em uma faixa onde os gradientes são significativos e o modelo consegue aprender efetivamente.

## Exemplo de Input/Output

### Matrizes de Entrada

**Q** (4×3) — Queries:
```
[[0.2, 0.8, 0.1],
 [0.9, 0.1, 0.5],
 [0.3, 0.6, 0.7],
 [0.5, 0.5, 0.0]]
```

**K** (3×3) — Keys:
```
[[0.6, 0.3, 0.4],
 [0.1, 0.9, 0.2],
 [0.7, 0.2, 0.8]]
```

**V** (3×3) — Values:
```
[[1.0, 0.5, 0.0],
 [0.0, 1.0, 0.5],
 [0.5, 0.0, 1.0]]
```

### Saída Esperada

Executando `python test_attention.py`, os testes validam que:

1. Cada linha dos **attention weights** soma exatamente 1.0 (distribuição de probabilidade válida)
2. Todos os attention weights são **não-negativos**
3. O **shape do output** é (4, 3) — número de queries × dimensão dos values
4. Os valores numéricos coincidem com o cálculo manual da fórmula

```
==================================================
TESTS
==================================================
  test_weights_sum_to_one: PASSED
  test_weights_non_negative: PASSED
  test_output_shape: PASSED (shape=(4, 3))
  test_numerical_correctness: PASSED

Result: 4/4 tests passed.
```

## Estrutura do Repositório

```
self-attention/
├── attention.py          # softmax e scaled_dot_product_attention
├── test_attention.py     # testes de validação numérica
├── requirements.txt      # dependências (numpy)
└── README.md
```

## Referência

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems, 30 (NIPS 2017).
