import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

matplotlib.use("Qt5Agg")

from algo_derivation1K import derivee_seconde_num
from simulation_schrodinger1K import generer_barriere_potentiel, generer_paquet_onde, calculer_transmission

def simuler_silencieux(Re: np.ndarray, Im: np.ndarray, V: np.ndarray, dx: float, dt: float, nt: int) -> np.ndarray:
    """
    Moteur de calcul sans affichage : propage Re/Im nt pas de temps et renvoie la densité finale.
    Utilise les mêmes équations que la version animée (algorithme de type Verlet / semi-implicite).
    """
    for _ in range(nt):
        d2_Re = derivee_seconde_num(Re, dx)
        Im = Im + (dt / 2.0) * d2_Re - dt * V * Re

        d2_Im = derivee_seconde_num(Im, dx)
        Re = Re - (dt / 2.0) * d2_Im + dt * V * Im

    return Re**2 + Im**2

if __name__ == "__main__":
    nx = 1000
    x, dx = np.linspace(-50.0, 50.0, nx, retstep=True)
    nt = 20000
    dt = 0.001

    k0_fixe = 2.8
    largeur_paquet = 2.0
    xc = -20.0
    debut_barriere = 5.0

    print("Calcul de l'influence de la largeur 'a' en cours...")
    V0_fixe = 10.0
    valeurs_a = np.linspace(1.0, 10.0, 30)
    T_a = []

    debut_chrono = time.time()
    for a_barriere in valeurs_a:
        V = generer_barriere_potentiel(x, debut_barriere, a_barriere, V0_fixe)
        psi_init = generer_paquet_onde(x, xc, k0_fixe, largeur_paquet)

        Re = np.real(psi_init)
        Im = np.imag(psi_init)
        densite_fin = simuler_silencieux(Re, Im, V, dx, dt, nt)

        T = calculer_transmission(x, dx, densite_fin, psi_init, debut_barriere + a_barriere)
        T_a.append(T)
        print(f"  - Test pour a = {a_barriere:.2f} terminé.")

    print(f"\nCalcul de l'influence de V0 en cours...")
    a_fixe = 5.0
    valeurs_V0 = np.linspace(1.0, 20.0, 30)
    T_V0 = []

    for V0_test in valeurs_V0:
        V = generer_barriere_potentiel(x, debut_barriere, a_fixe, V0_test)
        psi_init = generer_paquet_onde(x, xc, k0_fixe, largeur_paquet)

        Re = np.real(psi_init)
        Im = np.imag(psi_init)
        densite_fin = simuler_silencieux(Re, Im, V, dx, dt, nt)

        T = calculer_transmission(x, dx, densite_fin, psi_init, debut_barriere + a_fixe)
        T_V0.append(T)
        print(f"  - Test pour V0 = {V0_test:.2f} terminé.")

    fin_chrono = time.time()
    print(f"\nTemps total de calcul : {fin_chrono - debut_chrono:.1f} secondes.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(valeurs_a, T_a, marker='o', color='blue', linestyle='-')
    ax1.set_title("Influence de la largeur du barriere (a) sur la transmission")
    ax1.set_xlabel("Largeur du barriere (a)")
    ax1.set_ylabel("Coefficient de transmission T")
    ax1.grid(True)

    ax2.plot(valeurs_V0, T_V0, marker='s', color='red', linestyle='-')
    ax2.set_title("Influence de la profondeur du barriere (V0) sur la transmission")
    ax2.set_xlabel("Profondeur du barriere (V0)")
    ax2.set_ylabel("Coefficient de transmission T")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
