import numpy as np
import matplotlib.pyplot as plt

def f_carre(x: np.ndarray) -> np.ndarray:
    """Renvoie x**2 (vecteur)."""
    return x**2

def f_derivee_carre(x):
    """Renvoie la dérivée analytique de x**2 → 2x."""
    return 2*x

def f_derivee_seconde_carre(x):
    """Renvoie la dérivée seconde de x**2 → 2 (constante)."""
    return 2*np.ones_like(x)

def derivee_premiere_num(y: np.ndarray, dx: float) -> np.ndarray:
    """
    Approxime la dérivée première par différences centrées.
    Bords traités en forward/backward pour conserver la taille.
    """
    dy_num = np.zeros_like(y)
    dy_num[1:-1] = (y[2:] - y[:-2]) / (2 * dx)
    dy_num[0] = (y[1] - y[0]) / dx
    dy_num[-1] = (y[-1] - y[-2]) / dx
    return dy_num

def derivee_seconde_num(y: np.ndarray, dx: float) -> np.ndarray:
    """
    Approxime la dérivée seconde par différences finies centrées.
    Renvoie un tableau de même taille que y.
    """
    d2y_num = np.zeros_like(y)
    d2y_num[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2]) / (dx**2)
    d2y_num[0] = d2y_num[1]
    d2y_num[-1] = d2y_num[-2]
    return d2y_num

if __name__ == "__main__":
    npts = 1000
    x, dx = np.linspace(-5, 5, npts, retstep=True)

    y = f_carre(x)
    dy_exact = f_derivee_carre(x)
    dy_num = derivee_premiere_num(y, dx)

    erreur = np.abs(dy_exact - dy_num)
    print(f"Erreur maximale commise : {np.max(erreur):.5e}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(x, dy_exact, label="Analytique (2x)", color="blue")
    ax[0].plot(x, dy_num, label="Numérique", color="red", linestyle="--")
    ax[0].set_title("Comparaison des dérivées"); ax[0].legend()

    ax[1].plot(x, erreur, color="black"); ax[1].set_title("Erreur absolue de la dérivation")
    plt.tight_layout(); plt.savefig("Dérivée.png")

    d2y_exact = f_derivee_seconde_carre(x)
    d2y_num = derivee_seconde_num(y, dx)
    erreur_seconde = np.abs(d2y_exact - d2y_num)

    print(f"Erreur maximale commise (dérivée seconde) : {np.max(erreur_seconde):.5e}")

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
    ax2[0].plot(x, d2y_exact, label="Analytique (constante = 2)", color="blue")
    ax2[0].plot(x, d2y_num, label="Numérique", color="red", linestyle="--")
    ax2[0].set_title("Comparaison des dérivées secondes"); ax2[0].legend()

    ax2[1].plot(x, erreur_seconde, color="black"); ax2[1].set_title("Erreur absolue de la dérivation seconde")
    plt.tight_layout(); plt.savefig("Dérivée_seconde.png")
