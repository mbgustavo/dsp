import numpy as np
import matplotlib.pyplot as plt
from utils import (
    analisar,
    atrasar,
    autocorrelacao,
    coef_correlacao,
    correlacao,
    covariancia,
    desvio_padrao,
    gerar_x,
    media,
    media_movel,
    primeiro_lobulo,
    quantizar,
    variancia,
)

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
    R_ = autocorrelacao(gerar_x(n_per, f1, fs, A1, A2), K_comp)
    k_pico, freq, amp, Ps, Pn, snr = analisar(R_, fs)
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
    x_ = gerar_x(n_per, f1, fs, A1, A2)
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
R_400 = autocorrelacao(gerar_x(400, f1, fs, A1, A2), K_comp)
# tamanho da janela da media movel deve ser igual ao definido em analisar() para que os resultados coincidam
# tamanho da janel = 5
S_400 = media_movel(R_400, 5)
k_ini, k_fim = primeiro_lobulo(S_400)
k_pico, freq, amp, Ps, Pn, snr = analisar(R_400, fs)
 
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
