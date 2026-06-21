#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

matplotlib.use("Qt5Agg")

from algo_derivation import derivee_seconde_num
from simulation_schrodinger import generer_barriere_potentiel, generer_paquet_onde

def simuler_silencieux_2D(x: np.ndarray, dx: float, dt: float, nt: int, V: np.ndarray, psi_initial: np.ndarray) -> np.ndarray:
    """
    Propage psi_initial pendant nt pas et renvoie la matrice densité (nx, nt).
    Implémentation en stockage 2D : Re[:, n], Im[:, n].
    """
    nx = len(x)
    Re = np.zeros((nx, nt))
    Im = np.zeros((nx, nt))

    Re[:, 0] = np.real(psi_initial)
    Im[:, 0] = np.imag(psi_initial)

    for n in range(nt - 1):
        d2_Re = derivee_seconde_num(Re[:, n], dx)
        Im[:, n+1] = Im[:, n] + (dt / 2.0) * d2_Re - dt * V * Re[:, n]

        d2_Im = derivee_seconde_num(Im[:, n+1], dx)
        Re[:, n+1] = Re[:, n] - (dt / 2.0) * d2_Im + dt * V * Im[:, n+1]

    return Re**2 + Im**2

def calculer_temps_transit(x: np.ndarray, dt: float, densite_2D: np.ndarray, debut_b: float, largeur_a: float):
    """
    Renvoie (t_entree, t_sortie) mesurés via argmax de la densité aux indices de bord.
    """
    fin_b = debut_b + largeur_a
    idx_debut = np.argmin(np.abs(x - debut_b))
    idx_fin = np.argmin(np.abs(x - fin_b))

    t_entree = np.argmax(densite_2D[idx_debut, :]) * dt
    t_sortie = np.argmax(densite_2D[idx_fin, :]) * dt

    return t_entree, t_sortie

if __name__ == "__main__":
    nx = 1000
    x, dx = np.linspace(-50.0, 50.0, nx, retstep=True)
    nt = 20000
    dt = 0.001

    k0_fixe = 2.0
    largeur_paquet = 8.0
    xc = -30.0
    debut_barriere = 5.0

    print("Démarrage de l'étude temporelle...")
    debut_chrono = time.time()

    V0_fixe = 2.5
    valeurs_a = np.linspace(0.5, 4.0, 10)
    liste_tau_0 = []
    liste_tau_t_a = []

    for a_test in valeurs_a:
        psi_init = generer_paquet_onde(x, xc, k0_fixe, largeur_paquet)

        V_libre = np.zeros_like(x)
        H_libre = simuler_silencieux_2D(x, dx, dt, nt, V_libre, psi_init)
        t_in_libre, t_out_libre = calculer_temps_transit(x, dt, H_libre, debut_barriere, a_test)
        liste_tau_0.append(t_out_libre - t_in_libre)

        V_barriere = generer_barriere_potentiel(x, debut_barriere, a_test, V0_fixe)
        H_barriere = simuler_silencieux_2D(x, dx, dt, nt, V_barriere, psi_init)
        _, t_out_barriere = calculer_temps_transit(x, dt, H_barriere, debut_barriere, a_test)

        liste_tau_t_a.append(t_out_barriere - t_in_libre)

    a_fixe = 1.0
    valeurs_V0 = np.linspace(2.5, 6, 10)
    liste_tau_t_V0 = []

    for V0_test in valeurs_V0:
        psi_init = generer_paquet_onde(x, xc, k0_fixe, largeur_paquet)

        V_libre = np.zeros_like(x)
        H_libre = simuler_silencieux_2D(x, dx, dt, nt, V_libre, psi_init)
        t_in_libre, _ = calculer_temps_transit(x, dt, H_libre, debut_barriere, a_fixe)

        V_barriere = generer_barriere_potentiel(x, debut_barriere, a_fixe, V0_test)
        H_barriere = simuler_silencieux_2D(x, dx, dt, nt, V_barriere, psi_init)
        _, t_out_barriere = calculer_temps_transit(x, dt, H_barriere, debut_barriere, a_fixe)
        liste_tau_t_V0.append(t_out_barriere - t_in_libre)

    temps_total = time.time() - debut_chrono
    print(f"\nCalculs terminés en {temps_total:.1f} secondes. Affichage des graphes.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(valeurs_a, liste_tau_0, marker='o', label="Particule libre τ₀,num", color="green")
    ax1.plot(valeurs_a, liste_tau_t_a, marker='s', label="Effet tunnel τt,num", color="blue")
    ax1.set_title("Influence de la largeur de la barrière (a)")
    ax1.set_xlabel("Largeur de la barrière (a)"); ax1.set_ylabel("Temps de parcours (s)")
    ax1.legend(); ax1.grid(True)

    ax2.plot(valeurs_V0, liste_tau_t_V0, marker='d', color="red", label="τt,num")
    ax2.set_title("Influence de la hauteur de la barrière (V0)")
    ax2.set_xlabel("Hauteur de la barrière (V0)"); ax2.set_ylabel("Temps de franchissement (s)")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout(); plt.show()
