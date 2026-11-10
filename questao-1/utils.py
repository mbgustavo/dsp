"""Funções auxiliares para análise de sinais da questão 1."""

import numpy as np


def media(x):
    """Calcula a média aritmética das amostras.

    Parâmetros:
        x: Sequência não vazia de amostras.

    Retorno:
        Média aritmética das amostras.
    """
    N = len(x)
    soma = 0.0
    for amostra in x:
        soma = soma + amostra
    return soma / N


def variancia(x):
    """Calcula a variância populacional das amostras.

    Parâmetros:
        x: Sequência não vazia de amostras.

    Retorno:
        Variância, calculada dividindo a soma dos desvios quadráticos por N.
    """
    N = len(x)
    m = media(x)
    soma_quadrados = 0.0
    for amostra in x:
        soma_quadrados = soma_quadrados + (amostra - m) ** 2
    return soma_quadrados / N


def desvio_padrao(x):
    """Calcula o desvio padrão populacional de uma sequência.

    Parâmetros:
        x: Sequência não vazia de amostras.

    Retorno:
        Raiz quadrada da variância populacional.
    """
    return variancia(x) ** 0.5


def correlacao(a, b):
    """Calcula a soma dos produtos de amostras correspondentes.

    Parâmetros:
        a: Primeira sequência de amostras.
        b: Segunda sequência, com pelo menos o mesmo tamanho de `a`.

    Retorno:
        Soma de a[n] * b[n] para cada índice de `a`.
    """
    soma = 0.0
    for n in range(len(a)):
        soma = soma + a[n] * b[n]
    return soma


def autocorrelacao(x, K):
    """Calcula a autocorrelação de x para K atrasos, com normalização por sobreposição.

    Parâmetros:
        x: Sequência de amostras do sinal.
        K: Número de atrasos a calcular, começando em zero.

    Retorno:
        Array NumPy de K valores de autocorrelação; R[0] é a potência média.
    """
    N = len(x)
    R = np.zeros(K)
    for k in range(K):
        R[k] = correlacao(x, atrasar(x, k)) / (N - k)
    return R


def covariancia(a, b):
    """Calcula a covariância populacional entre duas sequências.

    Parâmetros:
        a: Primeira sequência de amostras.
        b: Segunda sequência, do mesmo tamanho de `a`.

    Retorno:
        Média dos produtos dos desvios em relação às respectivas médias.
    """
    N = len(a)
    ma = media(a)
    mb = media(b)
    soma = 0.0
    for n in range(N):
        soma = soma + (a[n] - ma) * (b[n] - mb)
    return soma / N


def coef_correlacao(a, b):
    """Calcula o coeficiente de correlação de Pearson entre duas sequências.

    Parâmetros:
        a: Primeira sequência de amostras.
        b: Segunda sequência, do mesmo tamanho de `a`.

    Retorno:
        Covariância normalizada pelos desvios padrão.
    """
    return covariancia(a, b) / (desvio_padrao(a) * desvio_padrao(b))


def atrasar(x, k):
    """Atrasa um sinal k amostras, preenchendo o início com zeros.

    Parâmetros:
        x: Sequência de amostras do sinal.
        k: Número de amostras de atraso.

    Retorno:
        Array NumPy do mesmo tamanho, deslocado à direita.
    """
    N = len(x)
    y = np.zeros(N)
    for n in range(k, N):
        y[n] = x[n - k]
    return y


def quantizar(x, n_bits, v_min, v_max):
    """Quantiza amostras uniformemente após saturá-las ao intervalo informado.

    Parâmetros:
        x: Sequência de amostras a quantizar.
        n_bits: Número de bits do quantizador.
        v_min: Limite inferior representável.
        v_max: Limite superior representável.

    Retorno:
        Tupla (xq, delta), com o sinal quantizado e o passo entre níveis.
    """
    niveis = 2 ** n_bits
    delta = (v_max - v_min) / (niveis - 1)
    xq = np.zeros(len(x))
    for i in range(len(x)):
        v = x[i]
        if v > v_max:
            v = v_max
        if v < v_min:
            v = v_min
        codigo = int((v - v_min) / delta + 0.5)
        xq[i] = v_min + codigo * delta
    return xq, delta


def media_movel(R, w):
    """Suaviza uma sequência usando a média local de uma janela centrada.

    Parâmetros:
        R: Sequência de valores a suavizar.
        w: Largura da janela em amostras.

    Retorno:
        Array NumPy suavizado, do mesmo tamanho de R.
    """
    meio = w // 2
    S = np.zeros(len(R))
    for k in range(len(R)):
        ini = max(0, k - meio)
        fim = min(len(R), k + meio + 1)
        S[k] = media(R[ini:fim])
    return S


def primeiro_lobulo(S):
    """Encontra os limites do primeiro lóbulo positivo após o índice zero.

    Parâmetros:
        S: Sequência suavizada de autocorrelação, contendo o lóbulo completo.

    Retorno:
        Tupla (k_ini, k_fim), índices que delimitam o lóbulo [k_ini, k_fim).
    """
    k = 1
    while S[k] > 0:
        k = k + 1
    while S[k] <= 0:
        k = k + 1
    k_ini = k
    while S[k] > 0:
        k = k + 1
    k_fim = k
    return k_ini, k_fim


def analisar(R, fs, janela=5):
    """Estima frequência, amplitude, potências e SNR a partir da autocorrelação.

    Parâmetros:
        R: Sequência de autocorrelação do sinal.
        fs: Taxa de amostragem em hertz.
        janela: Largura da janela usada para suavizar R, em amostras.

    Retorno:
        Tupla (k_pico, freq, amplitude, P_sinal, P_ruido, snr_db), com o período
        em amostras, frequência em Hz, amplitude e potências, e SNR em dB.
    """
    S = media_movel(R, janela)
    k_ini, k_fim = primeiro_lobulo(S)
    k_pico = (k_ini + k_fim) / 2
    periodo = k_pico / fs
    freq = 1 / periodo
    P_sinal = max(S[k_ini:k_fim])
    amplitude = (2 * P_sinal) ** 0.5
    P_total = R[0]
    P_ruido = P_total - P_sinal
    snr_db = 10 * np.log10(P_sinal / P_ruido)
    return k_pico, freq, amplitude, P_sinal, P_ruido, snr_db


def gerar_x(n_periodos, f1, fs, A1, A2):
    """Gera senoide somada a ruído uniforme reproduzível.

    Parâmetros:
        n_periodos: Duração do sinal em períodos da senoide.
        f1: Frequência da senoide em Hz.
        fs: Taxa de amostragem em Hz.
        A1: Amplitude de pico da senoide.
        A2: Amplitude máxima do ruído uniforme, distribuído entre -A2 e A2.

    Retorno:
        Array NumPy com a soma da senoide e do ruído.
    """
    t_ = np.arange(0, n_periodos / f1, 1 / fs)
    x1_ = A1 * np.sin(2 * np.pi * f1 * t_)
    rng_ = np.random.default_rng(seed=42)
    x2_ = rng_.uniform(-A2, A2, size=t_.size)
    return x1_ + x2_
