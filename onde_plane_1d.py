import numpy as np
import matplotlib.pyplot as plt

def PlaneWave(amp: float, k: float, omega: float, x: np.ndarray, t: float) -> np.ndarray:
    """
    amp: float,
    k: float,
    omega: float,
    x: np.ndarray,
    t: float,

    -----

    Retourne la fonction onde plane sous la forme Ae^i(kx - wt)
    avec    A = amp
            e = np.exp 
            i = ij
            w = omega
    """
    return amp * np.exp(1j * (k * x - omega * t))


if __name__ == "__main__":
    x = np.linspace(-10, 10, 500)
    psi = PlaneWave(1.0, 2.0, 1.0, x, 0.0)

    # Partie réelle de l'onde
    psi_re = np.real(psi)
    # Partie imaginaire de l'onde
    psi_im = np.imag(psi)

    fig, ax = plt.subplots()
    ax.plot(x, psi_re, label="Partie Réelle (cos)", color="blue")
    ax.plot(x, psi_im, label="Partie Imaginaire (sin)", color="red", linestyle="--")
    ax.set_title("Onde plane à 1D à l'instant t = 0")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()
    
    plt.show()
