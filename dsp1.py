import numpy as np
import matplotlib.pyplot as plt

def media(x):
    # Soma todas as amostras e divide por N
    N = len(x)
    soma = 0.0
    for amostra in x:
        soma = soma + amostra
    return soma / N

def variancia(x):
    # Média dos quadrados das distâncias até a média
    N = len(x)
    m = media(x)                 # reaproveita a função de cima
    soma_quadrados = 0.0
    for amostra in x:
        soma_quadrados = soma_quadrados + (amostra - m) ** 2
    return soma_quadrados / N

def desvio_padrao(x):
    # Raiz quadrada da variância
    return variancia(x) ** 0.5

def correlacao(a, b):
    # Multiplica amostra por amostra e soma tudo
    soma = 0.0
    for n in range(len(a)):
        soma = soma + a[n] * b[n]
    return soma

def autocorrelacao(x, K):
    # Para cada atraso k, correlaciona o sinal com ele mesmo atrasado.
    # Divide por N para o resultado ficar em V² (R[0] = potência média).
    N = len(x)
    R = np.zeros(K)
    for k in range(K):
        R[k] = correlacao(x, atrasar(x, k)) / (N - k)
    return R

# ---------------- Covariância ----------------
# nao usado no exercicio
def covariancia(a, b):
    # Correlação dos sinais em relação à média, dividida por N
    N = len(a)
    ma = media(a)
    mb = media(b)
    soma = 0.0
    for n in range(N):
        soma = soma + (a[n] - ma) * (b[n] - mb)
    return soma / N

# ---------------- Coeficiente de correlação ----------------
# nao usado no exercicio
def coef_correlacao(a, b):
    # Covariância normalizada pelos desvios padrão -> resultado entre -1 e +1
    return covariancia(a, b) / (desvio_padrao(a) * desvio_padrao(b))

# ---------------- Atraso de um sinal ----------------
def atrasar(x, k):
    # Desloca o sinal k amostras para a direita.
    # As k primeiras posições ficam com zero e as k últimas
    # amostras do original "saem" pelo fim -> mesmo tamanho.
    N = len(x)
    y = np.zeros(N)
    #range(k, N) -> começa em k (em vez de 0) e vai até N-1
    for n in range(k, N):
        y[n] = x[n - k]
    return y

def quantizar(x, n_bits, v_min, v_max):
    niveis = 2 ** n_bits                     # número de níveis possíveis
    delta = (v_max - v_min) / (niveis - 1)   # passo de quantização (V)
    xq = np.zeros(len(x))                    # vetor de saída
    for i in range(len(x)):
        v = x[i]
        # 1) satura: valores fora da faixa ficam no limite
        if v > v_max:
            v = v_max
        if v < v_min:
            v = v_min
        # 2) descobre o código digital (0, 1, 2, ..., niveis-1)
        codigo = int((v - v_min) / delta + 0.5)   # +0.5 arredonda
        # 3) converte o código de volta para tensão
        xq[i] = v_min + codigo * delta
    return xq, delta

# ---------------- Extração de frequência, amplitude e potência ----------------
def media_movel(R, w):
    # Cada ponto vira a média dele com os vizinhos (janela de w pontos).
    # Atenua o "serrilhado" do ruído sem deformar a cossenoide lenta.
    meio = w // 2
    S = np.zeros(len(R))
    for k in range(len(R)):
        ini = max(0, k - meio)
        fim = min(len(R), k + meio + 1)
        S[k] = media(R[ini:fim])
    return S

def primeiro_lobulo(S):
    # Encontra o 1º lóbulo positivo depois de k = 0 (o pico do 1º período)
    k = 1
    while S[k] > 0:          # 1) desce do pico de k = 0 até ficar negativo
        k = k + 1
    while S[k] <= 0:         # 2) atravessa o vale negativo
        k = k + 1
    k_ini = k                # 3) início do lóbulo positivo
    while S[k] > 0:          # 4) percorre o lóbulo até voltar a ser negativo
        k = k + 1
    k_fim = k
    return k_ini, k_fim

def analisar(R):
    # tamanho da janela da media movel tambem deve ser atualizado ao plotar grafico lah embaixo.
    # tamanho da janel = 5
    S = media_movel(R, 5)
    k_ini, k_fim = primeiro_lobulo(S)
    k_pico = (k_ini + k_fim) / 2           # centro do lóbulo = 1 período (amostras)
    periodo = k_pico / fs                  # s
    freq = 1 / periodo                     # Hz
    P_sinal = max(S[k_ini:k_fim])          # altura do pico = A²/2  (V²)
    amplitude = (2 * P_sinal) ** 0.5       # A = raiz(2·P)  (V)
    P_total = R[0]                         # potência de x (senoide + ruído), sem suavizar
    P_ruido = P_total - P_sinal
    snr_db = 10 * np.log10(P_sinal / P_ruido)
    return k_pico, freq, amplitude, P_sinal, P_ruido, snr_db

# ---------------- Gerar sinal com diferentes períodos ----------------
def gerar_x(n_periodos):
    # Gera o sinal completo x = x1 + x2 com a duração pedida
    t_ = np.arange(0, n_periodos / f1, 1 / fs)
    x1_ = A1 * np.sin(2 * np.pi * f1 * t_)
    rng_ = np.random.default_rng(seed=42)
    x2_ = rng_.uniform(-A2, A2, size=t_.size)
    return x1_ + x2_

# ---------------- Amostragem ----------------
##### DEFINICAO DE DURAÇÃO DO SINAL EM PERÍODOS DA SENOIDE
f1 = 17                    # frequência da senoide (Hz)
duracao = 40 / f1           # x períodos da senoide

fs = 1000                  # taxa de amostragem (Hz)
t = np.arange(0, duracao, 1 / fs) # cria vetor de tempo 

# ---------------- Sinal x1: senoide 1 Vpp, 17 Hz ----------------
Vpp1 = 1.0
A1 = Vpp1 / 2
x1 = A1 * np.sin(2 * np.pi * f1 * t)

# ---------------- Sinal x2: ruído aleatório 5 Vpp ----------------
# Distribuição uniforme entre -2,5 V e +2,5 V -> pico a pico = 5 V
# conferir se ruido esta certo, ruido gaussiano rever calculos de potencia
Vpp2 = 5.0
A2 = Vpp2 / 2
rng = np.random.default_rng(seed=42)   # semente fixa: resultado reproduzível
x2 = rng.uniform(-A2, A2, size=t.size)

print(f"x2 -> mín: {x2.min():.3f} V | máx: {x2.max():.3f} V | "
      f"Vpp medido: {x2.max() - x2.min():.3f} V | média: {x2.mean():.3f} V")

# ---------------- Sinal x: x1 + x2 ----------------
x = x1 + x2

"""# ---------------- Calculo com funções prontas ----------------
media_x1 = np.mean(x1)
var_x1 = np.var(x1)
media_x2 = np.mean(x2)
var_x2 = np.var(x2)
dp_x1 = np.std(x1)
dp_x2 = np.std(x2)

#print(f'----------Cálculo com funções prontas:-------')
#print(f"x1 -> média: {media_x1:.4f} V | variância: {var_x1:.4f} V² | desvio padrão: {dp_x1:.4f} V")
#print(f"x2 -> média: {media_x2:.4f} V | variância: {var_x2:.4f} V² | desvio padrão: {dp_x2:.4f} V")
"""
# ---------------- Calculo "na unha" ----------------
media_x1_m = media(x1)
var_x1_m = variancia(x1)
media_x2_m = media(x2)
var_x2_m = variancia(x2)
dp_x1 = desvio_padrao(x1)
dp_x2 = desvio_padrao(x2)

#print(f"\n---------Cálculo 'na unha':------------")
print(f"x1 -> média: {media_x1_m:.4f} V | variância: {var_x1_m:.4f} V² | desvio padrão: {dp_x1:.4f} V")
print(f"x2 -> média: {media_x2_m:.4f} V | variância: {var_x2_m:.4f} V² | desvio padrão: {dp_x2:.4f} V")

# ---------------- Quantizar ----------------
"""n_bits = 6
x1q, delta = quantizar(x1, n_bits, -A1, A1)
erro_q = x1 - x1q

print(f"Quantização de x1 com {n_bits} bits: {2**n_bits} níveis, passo = {delta:.4f} V")
#print(f"Níveis usados: {np.unique(x1q)}")
print(f"Erro máximo: {np.max(np.abs(erro_q)):.4f} V (teórico: delta/2 = {delta/2:.4f} V)")
"""

# ---------------- Correlação entre dois sinais ----------------
print(f"\ncorrelação x1 com x1: {correlacao(x1, x1):.4f}")
print(f"correlação x1 com x2: {correlacao(x1, x2):.4f}")
print(f"correlação x1 com x : {correlacao(x1, x):.4f}")

print(f"\ncovariância x1 com x1: {covariancia(x1, x1):.4f} V²  | variância x1: {variancia(x1):.4f} V²")
print(f"covariância x1 com x2: {covariancia(x1, x2):.4f} V²")
print(f"covariância x1 com x : {covariancia(x1, x):.4f} V²")

print(f"\ncoef. correlação x1 com x1: {coef_correlacao(x1, x1):.4f}  | numpy: {np.corrcoef(x1, x1)[0, 1]:.4f}")
print(f"coef. correlação x1 com x2: {coef_correlacao(x1, x2):.4f}  | numpy: {np.corrcoef(x1, x2)[0, 1]:.4f}")
print(f"coef. correlação x1 com x : {coef_correlacao(x1, x):.4f}  | numpy: {np.corrcoef(x1, x)[0, 1]:.4f}")
print(f"coef. correlação x1 com -x1: {coef_correlacao(x1, -x1):.4f}")
print(f"coef. correlação x1 com 10*x1: {coef_correlacao(x1, 10*x1):.4f}")

# ---------------- Autocorrelação ----------------
K = len(x) // 2                  # atrasos de 0 até metade do sinal
lags_ms = np.arange(K) / fs * 1000

R_x1 = autocorrelacao(x1, K)
R_x2 = autocorrelacao(x2, K)
R_x = autocorrelacao(x, K)

print(f"\nR_x1[0] = {R_x1[0]:.4f} V² | R_x2[0] = {R_x2[0]:.4f} V² | R_x[0] = {R_x[0]:.4f} V²")


# ---------------- Extração de frequência, amplitude e potência ----------------
# Valores verdadeiros, calculados a partir dos parâmetros dos sinais
P_sinal_teo = A1 ** 2 / 2                  # potência de uma senoide: A²/2
P_ruido_teo = Vpp2 ** 2 / 12               # variância da uniforme: (b - a)²/12
snr_teo = 10 * np.log10(P_sinal_teo / P_ruido_teo)
print(f"\nValores verdadeiros: f = {f1} Hz | A = {A1} V ({Vpp1} Vpp) | "
      f"P_sinal = {P_sinal_teo:.3f} V² | P_ruido = {P_ruido_teo:.3f} V² | "
      f"SNR = {snr_teo:.1f} dB")

#print("\nValores verdadeiros: f = 17 Hz | A = 0.5 V (1 Vpp) | P_sinal = 0.125 V² "
#      "| P_ruido = 2.083 V² | SNR = -12.2 dB")

K_comp = 200                          # mesmo intervalo de atrasos nos três (0 a 116 ms)
lags_comp = np.arange(K_comp) / fs * 1000
R_teorico = (A1 ** 2 / 2) * np.cos(2 * np.pi * f1 * lags_comp / 1000)   # gabarito

for n_per in [4, 40, 400]:
    R_ = autocorrelacao(gerar_x(n_per), K_comp)
    k_pico, freq, amp, Ps, Pn, snr = analisar(R_)
    print(f"{n_per:4d} períodos -> pico em {k_pico:.1f} ms | f = {freq:.2f} Hz | "
          f"A = {amp:.3f} V ({2*amp:.2f} Vpp) | P_sinal = {Ps:.3f} V² | "
          f"P_ruido = {Pn:.3f} V² | SNR = {snr:.1f} dB")

# ---------------- Gráficos ----------------
##########################

# ---------------- Gráficos dos sinais x1, x2 e x ----------------
amp = A1 + A2
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

ax1.plot(t * 1000, x1, color="tab:blue", label=f"x1: {Vpp1} Vpp, {f1} Hz")
ax1.set_title("Sinal x1 (senoide)")
ax1.set_ylabel("Amplitude (V)")
ax1.set_ylim(-amp, amp)                     # mesma escala dos dois para comparar
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right")

ax2.plot(t * 1000, x2, color="tab:orange", linewidth=0.8,
         label=f"x2: ruído uniforme {Vpp2} Vpp")
ax2.axhline(A2, color="gray", linestyle="--", linewidth=0.8)
ax2.axhline(-A2, color="gray", linestyle="--", linewidth=0.8)
ax2.set_title("Sinal x2 (ruído aleatório)")
ax2.set_ylabel("Amplitude (V)")
ax2.set_ylim(-amp, amp)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")

ax3.plot(t * 1000, x, color="tab:green", label=f"x: x1+x2")
ax3.set_title("Sinal x (x1 + x2)")
ax3.set_xlabel("Tempo (ms)")
ax3.set_ylabel("Amplitude (V)")
ax3.set_ylim(-amp, amp)
ax3.grid(True, alpha=0.3)
ax3.legend(loc="upper right")
ax3.set_xlabel("Tempo (ms)")

plt.tight_layout()
plt.savefig("sinais_x1_x2.png", dpi=150)

"""#quantizado
plt.figure(figsize=(10, 4))
plt.plot(t * 1000, x1, label="x1 original (float64)")
plt.step(t * 1000, x1q, where="mid", label=f"x1 quantizado ({n_bits} bits)")
plt.title("Quantização de x1")
plt.xlabel("Tempo (ms)")
plt.ylabel("Amplitude (V)")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("x1_quantizado.png", dpi=150)
"""

"""# Atrasando x1 em 15 amostras (15 ms)
k = 15
x1_atrasado = atrasar(x1, k)

plt.figure(figsize=(10, 4))
plt.plot(t * 1000, x1, label="x1")
plt.plot(t * 1000, x1_atrasado, label=f"x1 atrasado {k} amostras")
plt.title("Atraso de um sinal")
plt.xlabel("Tempo (ms)")
plt.ylabel("Amplitude (V)")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("x1_atrasado.png", dpi=150)
"""
# ---------------- Autocorrelação ----------------
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for ax, R, nome, cor in zip(axs, [R_x1, R_x2, R_x], ["x1 (senoide)", "x2 (ruído)", "x = x1 + x2"],
                            ["tab:blue", "tab:orange", "tab:green"]):
    ax.plot(lags_ms, R, color=cor)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"Autocorrelação de {nome}")
    ax.set_ylabel("R(k) (V²)")
    ax.grid(True, alpha=0.3)
axs[-1].set_xlabel("Atraso (ms)")
plt.tight_layout()
plt.savefig("autocorrelacao.png", dpi=150)

# ---------------- Gráficos autocorrelacao  ----------------
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
for ax, n_per in zip(axs, [4, 40, 400]):
    x_ = gerar_x(n_per)
    R_ = autocorrelacao(x_, K_comp)
    ax.plot(lags_comp, R_, color="tab:green", label="autocorrelação de x")
    ax.plot(lags_comp, R_teorico, color="tab:blue", linestyle="--", label="senoide pura (teórico)")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"{n_per} períodos (N = {len(x_)} amostras)")
    ax.set_ylabel("R(k) (V²)")
    ax.set_ylim(-0.5, 0.5)            # corta o pico de k = 0 para ver os detalhes
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
axs[-1].set_xlabel("Atraso (ms)")
plt.tight_layout()
plt.savefig("autocorrelacao_4_40_400.png", dpi=150)

# ---------------- Conferência visual (400 períodos) ----------------
# Obs.: media_movel e primeiro_lobulo já são chamadas dentro de analisar(),
# mas lá os resultados intermediários (S, k_ini, k_fim) não são devolvidos.
# Recalculo aqui só para poder plotá-los.
R_400 = autocorrelacao(gerar_x(400), K_comp)
# tamanho da janela da media movel deve ser igual ao definido em analisar() para que os resultados coincidam
# tamanho da janel = 5
S_400 = media_movel(R_400, 5)
k_ini, k_fim = primeiro_lobulo(S_400)
k_pico, freq, amp, Ps, Pn, snr = analisar(R_400)
 
plt.figure(figsize=(10, 4))
plt.plot(lags_comp, R_400, color="tab:green", alpha=0.4, label="R(k) de x")
plt.plot(lags_comp, S_400, color="tab:green", label="R(k) suavizado (média móvel 5)")
plt.axvspan(k_ini, k_fim, color="gold", alpha=0.3, label="1º lóbulo positivo")
plt.axvline(k_pico, color="red", linestyle="--", label=f"período = {k_pico:.0f} ms → f = {freq:.2f} Hz")
plt.axhline(Ps, color="purple", linestyle=":", label=f"P_sinal = {Ps:.3f} V² → A = {amp:.3f} V")
plt.axhline(0, color="gray", linewidth=0.8)
plt.ylim(-0.3, 0.4)
plt.title("Detecção da senoide pela autocorrelação (400 períodos)")
plt.xlabel("Atraso (ms)")
plt.ylabel("R(k) (V²)")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("deteccao_400.png", dpi=150)

#------------------
plt.show()
