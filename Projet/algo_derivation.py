import numpy as np
import matplotlib.pyplot as plt

def f_carre(x: np.ndarray) -> np.ndarray:
    """
    Retourne le carré de chaque élément du tableau x
    """
    return x**2

def f_derivee_carre(x):
    """
    Retourne la dérivée de la fonction x^2 => 2x
    """
    return 2*x

def derivee_premiere_num(y: np.ndarray, dx: float) -> np.ndarray:
    """
    Calcule la dérivée première
    """
    dy_num = np.zeros_like(y)
    dy_num[1:-1] = (y[2:] - y[:-2]) / (2 * dx)

    # Gestion bords
    dy_num[0] = (y[1] - y[0]) / dx
    dy_num[-1] = (y[-1] - y[-2]) / dx
    return dy_num

def derivee_seconde_num(y: np.ndarray, dx: float) -> np.ndarray:
    """
    Calcule la dérivée seconde du tableau
    """
    d2y_num = np.zeros_like(y)
    d2y_num[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2]) / (dx**2)

    # Bords
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
    ax[0].plot(x, dy_num, label="Numérique", color="red",linestyle="--")
    ax[0].set_title("Comparaison des dérivées")
    ax[0].legend()

    ax[1].plot(x, erreur, color="black")
    ax[1].set_title("Erreur absolue de la dérivation")

    plt.tight_layout()
    plt.savefig("Dérivée.png")
