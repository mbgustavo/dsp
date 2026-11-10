import numpy as np

def correlacao_freq(x, fs, f):
    """Mede a magnitude da correlação de x com seno e cosseno na frequência f.

    Parâmetros:
        x: Sequência de amostras do sinal.
        fs: Taxa de amostragem em Hz.
        f: Frequência de referência em Hz.

    Retorno:
        Magnitude não normalizada da correlação, em unidades de x somadas.
    """
    # Correlaciona x com um cosseno e um seno de frequência f
    # e junta os dois (Pitágoras), para não depender da fase.
    C = 0.0
    D = 0.0
    for n in range(len(x)):
        angulo = 2 * np.pi * f * n / fs
        C = C + x[n] * np.cos(angulo)
        D = D + x[n] * np.sin(angulo)
    return (C ** 2 + D ** 2) ** 0.5
 
def dft_amplitude(x, fs):
    """Calcula o espectro de amplitude nas frequências até Nyquist.

    Parâmetros:
        x: Sequência de amostras do sinal.
        fs: Taxa de amostragem em Hz.

    Retorno:
        Tupla (freqs, amplitude) com frequências em Hz e amplitudes estimadas.
    """
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
 
def resposta_frequencia(h, fs, freqs):
    """Calcula o ganho de um filtro nas frequências especificadas.

    Parâmetros:
        h: Coeficientes da resposta ao impulso do filtro.
        fs: Taxa de amostragem em Hz.
        freqs: Sequência das frequências de avaliação, em Hz.

    Retorno:
        Array NumPy com o ganho calculado para cada frequência.
    """
    ganho = np.zeros(len(freqs))
    for i in range(len(freqs)):
        ganho[i] = correlacao_freq(h, fs, freqs[i])
    return ganho
 
