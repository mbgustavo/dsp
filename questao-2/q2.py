import numpy as np
import matplotlib.pyplot as plt

from shared import convolucao

np.set_printoptions(precision=3)

# ---------------- Sinal de entrada x(t) ----------------
T = 1                          # período de amostragem
t = np.arange(0, (32 + 1) * T, T)    # 0, 1, 2, ..., 32  (33 amostras)
x = np.exp(-0.5 * t)

# ---------------- Resposta ao impulso h(t) ----------------
h = np.array([0, 1.5, 0.75, 0.375, 0.1875, 0.09375, 0.046875, 0.0234375])
th = np.arange(len(h)) * T     # 0, 1, ..., 7

print(f"x: {len(x)} amostras | h: {len(h)} amostras")
print(f"x: {x}")

# ---------------- Convolução x(t) * h(t) ----------------
y = convolucao(x, h)
ty = np.arange(len(y)) * T               # 0, 1, ..., 39

print(f"y: {len(y)} amostras")
print(f"y: {y}")
#print(f"y (na unha), primeiras 8: {np.round(y[:8], 4)}")
#print(f"y (numpy),   primeiras 8: {np.round(np.convolve(x, h)[:8], 4)}")
#print(f"maior diferença entre os dois: {np.max(np.abs(y - np.convolve(x, h))):.2e}")

# ---------------- Gráficos ----------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

ax1.stem(t, x)
ax1.set_title("Entrada x(t) = exp(-0,5·t)")
ax1.set_ylabel("x(t)")
ax1.grid(True, alpha=0.3)

ax2.stem(th, h)
ax2.set_title("Resposta ao impulso h(t)")
ax2.set_ylabel("h(t)")
ax2.grid(True, alpha=0.3)

ax3.stem(ty, y)
ax3.set_title("Saída y(t) = x(t) ∗ h(t)")
ax3.set_xlabel("t (amostras)")
ax3.set_ylabel("y(t)")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("x_h_y.png", dpi=150)
plt.show()