import numpy as np
import matplotlib.pyplot as plt

# Unités réduites (hbar = m = 1) : cohérence avec les simulations numériques
hbar = 1.0
m = 1.0

def GaussWP(k0: float, a: float, x: np.ndarray, t: float) -> np.ndarray:
    """
    Paquet d'ondes gaussien analytique (cas général).
    - k0 : nombre d'onde central
    - a  : largeur (paramètre d'enveloppe)
    - x  : positions (numpy array)
    - t  : temps

    Renvoie ψ(x,t) (complexe). La formule gère t=0 et t≠0.
    """
    if t == 0:
        prefacteur = (2.0 / (np.pi * a**2))**0.25
        enveloppe = np.exp(-(x**2) / (a**2))
        porteuse = np.exp(1j * k0 * x)
        return prefacteur * enveloppe * porteuse
    else:
        denom = m * a**2 + 2j * hbar * t
        terme1 = (1.0 / (8.0 * np.pi**3))**0.25
        terme2 = np.sqrt(4.0 * np.pi * m * a / denom)
        exposant = (m / 4.0) * (a**2 * k0 + 2j * x)**2 / denom - (a**2 * k0**2) / 4.0
        return terme1 * terme2 * np.exp(exposant)

if __name__ == "__main__":
    k0_val = 10.0
    a_val = 2.0
    x_array = np.linspace(-10, 10, 1000)
    psi_0 = GaussWP(k0_val, a_val, x_array, 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_array, np.real(psi_0), label="Partie réelle", color="blue", linewidth=1.5)
    ax.plot(x_array, np.imag(psi_0), label="Partie imaginaire", color="red", linestyle="--", linewidth=1.5)
    densite = np.abs(psi_0)**2
    ax.plot(x_array, densite, label="Densité |psi|²", color="black", linestyle=":", linewidth=2)
    ax.set_title("Paquet d'ondes gaussien à t = 0")
    ax.set_xlabel("Position x"); ax.set_ylabel("Amplitude")
    ax.grid(True); ax.legend()
    plt.tight_layout(); plt.savefig("paquet_onde.png")
