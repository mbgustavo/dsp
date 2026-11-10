import numpy as np

def convolucao(x, h):
    """Calcula a convolução linear completa de dois sinais discretos.

    Parâmetros:
        x: Sequência de amostras do sinal de entrada.
        h: Sequência de coeficientes da resposta ao impulso.

    Retorno:
        Array NumPy com N + M - 1 amostras da convolução.
    """
    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1)              # tamanho da saída: N + M - 1
    for n in range(N + M - 1):           # para cada instante da saída...
        soma = 0.0
        for k in range(M):               # ...percorre os coeficientes de h
            if 0 <= n - k < N:           # só usa amostras de x que existem
                soma = soma + h[k] * x[n - k]
        y[n] = soma
    return y
