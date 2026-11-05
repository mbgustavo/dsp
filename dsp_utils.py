import numpy as np
 
# ---------------- Convolução ----------------
def convolucao(x, h):
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
 
# ---------------- Correlação com uma senoide de frequência f ----------------
def correlacao_freq(x, fs, f):
    # Correlaciona x com um cosseno e um seno de frequência f
    # e junta os dois (Pitágoras), para não depender da fase.
    C = 0.0
    D = 0.0
    for n in range(len(x)):
        angulo = 2 * np.pi * f * n / fs
        C = C + x[n] * np.cos(angulo)
        D = D + x[n] * np.sin(angulo)
    return (C ** 2 + D ** 2) ** 0.5
 
# ---------------- Espectro de um sinal (DFT) ----------------
def dft_amplitude(x, fs):
    # Testa as frequências k·fs/N, de 0 até Nyquist,
    # e divide por N/2 para obter a amplitude de cada senoide.
    N = len(x)
    n_freqs = N // 2 + 1
    freqs = np.zeros(n_freqs)
    amplitude = np.zeros(n_freqs)
    for k in range(n_freqs):
        freqs[k] = k * fs / N
        amplitude[k] = correlacao_freq(x, fs, freqs[k]) / (N / 2)
    return freqs, amplitude
 
# ---------------- Resposta em frequência de um filtro ----------------
def resposta_frequencia(h, fs, freqs):
    # Testa as frequências pedidas, sem normalizar:
    # o resultado é o GANHO do filtro em cada frequência.
    ganho = np.zeros(len(freqs))
    for i in range(len(freqs)):
        ganho[i] = correlacao_freq(h, fs, freqs[i])
    return ganho
 