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

def DisplayOneWave(A: float = 1.0, k: float = 10.0, omega: float = 0.0, t: float = 0.0) -> None:
    """
    Teste la fonction PlaneWave pour une onde plane unique et affiche
    sa partie réelle et sa partie imaginaire en fonction de x, à l'instant t.
    """
    # Intervalle d'affichage : quelques longueurs d'onde
    x_min = -2 * np.pi / k
    x_max = 2 * np.pi / k
    x = np.linspace(x_min, x_max, 1000)

    psi = PlaneWave(A, k, omega, x, t)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, np.real(psi), label="Partie réelle Re(Ψ)", color="blue")
    ax.plot(x, np.imag(psi), label="Partie imaginaire Im(Ψ)", color="red", linestyle="--")

    ax.set_title(f"Onde plane unique Ψ(x,t) à t = {t}")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("onde_plane_unique.png")



def DisplayThreeWaves(A: float = 1.0, k0: float = 10.0, delta_k: float = 2.0) -> None:
    """
    Génére et affiche la superposition de trois ondes planes à t = 0
    """

    x_min = -np.pi / delta_k
    x_max = np.pi / delta_k
    x = np.linspace(x_min, x_max, 1000)

    # Génération des trois ondes t = 0 (omega = 0)
    psi_1 = PlaneWave(A, k0, 0.0, x, 0.0)
    psi_2 = PlaneWave(A / 2.0, k0 - delta_k / 2.0, 0.0, x, 0.0)
    psi_3 = PlaneWave(A / 2.0, k0 + delta_k / 2.0, 0.0, x, 0.0)

    # Supérposition des ondes
    psi_tot = psi_1 + psi_2 + psi_3

    enveloppe = A * (1 + np.cos((delta_k / 2.0) * x))

    # Graphique
    fig, ax = plt.subplots(figsize=(10, 6))

    # Trois ondes
    ax.plot(x, np.real(psi_1), label=f"Onde centrale (k={k0})", linestyle=":", color="gray", alpha=0.7)
    ax.plot(x, np.real(psi_2), label=f"Onde gauche (k={k0 - delta_k/2})", linestyle=":", color="cyan", alpha=0.7)
    ax.plot(x, np.real(psi_3), label=f"Onde droite (k={k0 + delta_k/2})", linestyle=":", color="magenta", alpha=0.7)

    # Onde totale
    ax.plot(x, np.real(psi_tot), label="Onde résultante (Somme)", color="blue", linewidth=1.5)
    ax.plot(x, enveloppe, label="Enveloppe (+)", color="red", linestyle="--", linewidth=2)
    ax.plot(x, -enveloppe, label="Enveloppe (-)", color="red", linestyle="--", linewidth=2)

    ax.set_title("Superposition de 3 ondes planes et apparition de l'enveloppe spatiale (t=0)")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize="small")

    plt.tight_layout()
    plt.savefig("superposition_ondes.png")



if __name__ == "__main__":
    DisplayOneWave(A=1.0, k=10.0, omega=0.0, t=0.0)
    DisplayThreeWaves(A=1.0, k0=10.0, delta_k=2.0)
