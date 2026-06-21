import numpy as np
import matplotlib.pyplot as plt

hbar = 1.054571817e-34
m = 9.1093837015e-31

def GaussWP(k0: float, a: float, x: np.ndarray, t: float) -> np.ndarray:
    """
    Retourne le paquet d'ondes gaussien psi(x, t)
    """
    if t == 0:
        prefacteur = (2.0 / (np.pi * a**2))**0.25
        enveloppe = np.exp(-(x**2) / (a**2))
        porteuse = np.exp(1j * k0 * x)

        return prefacteur * enveloppe * porteuse
    else :
        pass

if __name__ == "__main__":
    k0_val = 10.0
    a_val = 2.0

    x_array = np.linspace(-10, 10, 1000)

    psi_0 = GaussWP(k0_val, a_val, x_array, 0.0)

    # Graph
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_array, np.real(psi_0), label="Partie réelle", color="blue", linewidth=1.5)
    ax.plot(x_array, np.imag(psi_0), label="Partie imaginaire", color="red", linestyle="--", linewidth=1.5)
    densite = np.abs(psi_0)**2
    ax.plot(x_array, densite, label="Densité de probilité |psi|²", color="black", linestyle=":", linewidth=2)

    ax.set_title("paquet d'ondes gaussien à l'instant t = 0")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig("paquet_onde.png")
