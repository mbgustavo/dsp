import numpy as np
import matplotlib.pyplot as plt

# ---------------- DFT na unha ----------------
def dft_amplitude(x, fs):
    # Para cada frequência de teste, correlaciona x com um cosseno e um seno
    N = len(x)
    n_freqs = N // 2 + 1                 # de 0 Hz até Nyquist
    freqs = np.zeros(n_freqs)
    amplitude = np.zeros(n_freqs)
    for k in range(n_freqs):
        f = k * fs / N                   # frequência testada (Hz)
        C = 0.0                          # correlação com o cosseno
        D = 0.0                          # correlação com o seno
        for n in range(N):
            angulo = 2 * np.pi * f * n / fs
            C = C + x[n] * np.cos(angulo)
            D = D + x[n] * np.sin(angulo)
        freqs[k] = f
        amplitude[k] = (C ** 2 + D ** 2) ** 0.5 / (N / 2)   # amplitude da senoide
    return freqs, amplitude

# ---------------- DFT passo a passo em uma frequência ----------------
def media(v):
    return sum(v) / len(v)

def dft_passo_a_passo(x, fs, f):
    # Mesma conta da dft_amplitude, para UMA frequência f,
    # guardando os produtos e as somas acumuladas a cada amostra.
    N = len(x)
    ref_cos = np.zeros(N)
    ref_sin = np.zeros(N)
    prod_cos = np.zeros(N)
    prod_sin = np.zeros(N)
    C_acum = np.zeros(N)
    D_acum = np.zeros(N)
    C = 0.0
    D = 0.0
    for n in range(N):
        angulo = 2 * np.pi * f * n / fs
        ref_cos[n] = np.cos(angulo)
        ref_sin[n] = np.sin(angulo)
        prod_cos[n] = x[n] * ref_cos[n]
        prod_sin[n] = x[n] * ref_sin[n]
        C = C + prod_cos[n]
        D = D + prod_sin[n]
        C_acum[n] = C
        D_acum[n] = D
    amplitude = (C ** 2 + D ** 2) ** 0.5 / (N / 2)
    return ref_cos, ref_sin, prod_cos, prod_sin, C_acum, D_acum, amplitude

def grafico_dft(x, fs, f, n_zoom, arquivo):
    ref_cos, ref_sin, prod_cos, prod_sin, C_acum, D_acum, amp = dft_passo_a_passo(x, fs, f)
    n = np.arange(len(x))
    nz = n[:n_zoom]

    fig, axs = plt.subplots(3, 1, figsize=(10, 9))

    # 1) o sinal e a senoide de teste (só o começo)
    axs[0].plot(nz, x[:n_zoom], "o-", markersize=3, label="S")
    axs[0].plot(nz, ref_sin[:n_zoom], "o-", markersize=3, color="tab:red", label=f"sen de teste ({f} Hz)")
    axs[0].set_title(f"1) Sinal S e senoide de teste de {f} Hz (primeiras {n_zoom} amostras)")
    axs[0].legend(loc="upper right", fontsize=8)

    # 2) o produto amostra a amostra (só o começo)
    axs[1].plot(nz, prod_sin[:n_zoom], "o-", markersize=3, color="tab:purple", label="S · sen")
    axs[1].axhline(media(prod_sin), color="black", linestyle="--",
                   label=f"média de S·sen em todas as amostras = {media(prod_sin):.3f}")
    axs[1].set_title(f"2) Produto S · sen({f} Hz) amostra a amostra")
    axs[1].legend(loc="upper right", fontsize=8)

    # 3) as somas acumuladas, ao longo de TODAS as amostras
    axs[2].plot(n, C_acum, label="C (soma acumulada de S·cos)")
    axs[2].plot(n, D_acum, label="D (soma acumulada de S·sen)")
    axs[2].set_title(f"3) Somas acumuladas ao longo das {len(x)} amostras -> amplitude = {amp:.3f}")
    axs[2].set_xlabel("n (amostra)")
    axs[2].legend(loc="upper left", fontsize=8)

    for ax in axs:
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(arquivo, dpi=150)

# ---------------- Parâmetros ----------------
fs = 8000                      # frequência de amostragem (Hz)
duracao = 6                    # duração do sinal (s)
f1 = 750                       # Hz
f2 = 2500                      # Hz
f3 = 3250                      # Hz

# ---------------- Sinal S ----------------
t = np.arange(0, duracao, 1 / fs)        # 0 até 6 s (sem incluir o 6)
S = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + np.sin(2 * np.pi * f3 * t)

print(f"S: {len(S)} amostras | duração: {len(S) / fs} s")
print(f"Nyquist: {fs / 2} Hz -> maior frequência do sinal: {max(f1, f2, f3)} Hz")
print(f"Quantidade de períodos de cada senoide em 6 s: f1: {f1 * duracao} | f2: {f2 * duracao} | f3: {f3 * duracao}")

# ---------------- Senoides separadas e soma, em versão "contínua" ----------------
# Simula o sinal contínuo com uma amostragem muito mais fina (1 MHz),
# só para desenhar. As amostras reais de 8 kHz aparecem como pontos por cima.
fs_cont = 1_000_000                       # 1 MHz -> ~300 pontos por ciclo de 3250 Hz
janela = 0.004                            # 4 ms = um período da soma
t_cont = np.arange(0, janela, 1 / fs_cont)

s1_cont = np.sin(2 * np.pi * f1 * t_cont)
s2_cont = np.sin(2 * np.pi * f2 * t_cont)
s3_cont = np.sin(2 * np.pi * f3 * t_cont)
S_cont = s1_cont + s2_cont + s3_cont

# amostras reais (8 kHz) dentro da janela: 32
n_jan = int(janela * fs)                  
t_amostras = t[:n_jan]
s1 = np.sin(2 * np.pi * f1 * t_amostras)
s2 = np.sin(2 * np.pi * f2 * t_amostras)
s3 = np.sin(2 * np.pi * f3 * t_amostras)

# Versão "contínua" de S nos mesmos 10 ms (amostragem fina de 1 MHz, só para desenhar)
t_cont10 = np.arange(0, 0.010, 1 / fs_cont)  # 10 ms = 0,010 s
S_cont10 = (np.sin(2 * np.pi * f1 * t_cont10) + np.sin(2 * np.pi * f2 * t_cont10)
            + np.sin(2 * np.pi * f3 * t_cont10))

# ---------------- FFT ----------------
# FFT do NumPy, no sinal inteiro (48000 amostras)
# cria vetor de frequências (0, 1, 2, ..., fs/2) com resolução fs/N
freqs_fft = np.fft.rfftfreq(len(S), 1 / fs)


# utiliza apenasos valores da serie, porem sem informação da frequência, por isso é necessário criar o vetor de frequências
# passo 
amp_fft = np.abs(np.fft.rfft(S)) / (len(S) / 2)
print(f"\n tamanho FFT:     {len(amp_fft)} amostras | resolução: {fs / len(S):.3f} Hz")

# ---------------- Espectro de frequências (DFT na unha) ----------------
# utiliza menos amostras para conseguir rodar
N_trecho = 800      # 100 ms -> resolução de fs/N = 10 Hz
freqs_unha, amp_unha = dft_amplitude(S[:N_trecho], fs)


for f, a in zip(freqs_fft, amp_fft):
    if a > 0.1:#> 0.5:
        print(f"FFT:     {f:7.1f} Hz -> amplitude {a:.3f}")

# ---------------- Gráfico continuo ----------------
fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
curvas = [(s1_cont, s1, f"s1: {f1} Hz"),
          (s2_cont, s2, f"s2: {f2} Hz"),
          (s3_cont, s3, f"s3: {f3} Hz"),
          (S_cont, S[:n_jan], "S = s1 + s2 + s3")]
for ax, (cont, amostras, nome) in zip(axs, curvas):
    ax.plot(t_cont * 1000, cont, label="contínuo (1 MHz)", color="tab:blue")
    ax.plot(t_amostras * 1000, amostras, "o", color="tab:red", markersize=4, label="amostras (8 kHz)")
    ax.set_title(nome)
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
axs[0].legend(loc="upper right", fontsize=8)
axs[-1].set_xlabel("Tempo (ms)")
plt.tight_layout()
plt.savefig("senoides_continuas.png", dpi=150)

# ---------------- Gráfico (só os primeiros 10 ms) ----------------
n_mostrar = int(0.010 * fs)              # 10 ms = 80 amostras

plt.figure(figsize=(10, 4))
plt.plot(t_cont10 * 1000, S_cont10, color="tab:red", linewidth=1, label="S contínuo")
plt.plot(t[:n_mostrar] * 1000, S[:n_mostrar], marker="o", markersize=3,
         label="S amostrado (8 kHz), pontos ligados por retas")
plt.axvspan(4, 8, color="gold", alpha=0.2, label="1 período de S (4 ms)")
plt.legend(loc="upper right", fontsize=8)
plt.title("Sinal S = sen(2π·750·t) + sen(2π·2500·t) + sen(2π·3250·t)  (primeiros 10 ms)")
plt.xlabel("Tempo (ms)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sinal_S.png", dpi=150)

# ---------------- Gráfico FFT ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.stem(freqs_unha, amp_unha, markerfmt=" ")
ax1.set_title(f"Espectro de S: DFT na unha ({N_trecho} amostras, resolução {fs / N_trecho:.0f} Hz)")
ax1.set_ylabel("Amplitude")
ax1.grid(True, alpha=0.3)
ax2.plot(freqs_fft, amp_fft)
ax2.set_title(f"Espectro de S: FFT do NumPy ({len(S)} amostras, resolução {fs / len(S):.3f} Hz)")
ax2.set_xlabel("Frequência (Hz)")
ax2.set_ylabel("Amplitude")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("espectro_S.png", dpi=150)

grafico_dft(S[:800], fs, 750, 64, "dft_750Hz.png")
grafico_dft(S[:800], fs, 760, 64, "dft_760Hz.png")
grafico_dft(S[:800], fs, 800, 64, "dft_800Hz.png")
plt.show()