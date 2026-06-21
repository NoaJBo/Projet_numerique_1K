#!/usr/bin/env python3

"""
Validation 3.2.5 du sujet :
"Confronter les résultats de l'algorithme (dans le cas V0 = 0) avec le
programme PaquetOndes.py"

On compare ici la simulation numérique de l'équation de Schrödinger pour
une particule libre (V = 0, algorithme de simulation_schrodinger.py) à la
solution analytique exacte du paquet d'ondes gaussien GaussWP(x, t)
(paquet_onde_gauss_1d.py, éq. 5 du sujet), aux mêmes instants.

Si l'algorithme numérique est correct, les deux courbes doivent coïncider
(à l'erreur de discrétisation près) : c'est cette vérification qui valide
l'algorithme AVANT de l'utiliser avec une barrière de potentiel (partie 4).
"""

import numpy as np
import matplotlib.pyplot as plt

from algo_derivation import derivee_seconde_num
from paquet_onde_gauss_1d import GaussWP, hbar, m  # unités réduites hbar = m = 1


def generer_paquet_onde(x: np.ndarray, xc: float, k0: float, a: float) -> np.ndarray:
    """
    Identique à la fonction du même nom dans simulation_schrodinger.py.
    Redéfinie ici pour ne pas dépendre du backend Qt5Agg (inutile pour
    cette validation, qui ne fait pas d'animation).
    """
    prefacteur = (2.0 / (np.pi * a**2)) ** 0.25
    enveloppe = np.exp(-((x - xc)**2) / (a**2))
    porteuse = np.exp(1j * k0 * x)
    return prefacteur * enveloppe * porteuse


def propager_libre(Re: np.ndarray, Im: np.ndarray, dx: float, dt: float, nt: int):
    """
    Propage (Re, Im) de nt pas de temps avec l'algorithme de Verlet,
    pour une particule libre (V = 0). Mêmes équations que dans
    simulation_schrodinger.py / etude_parametres.py.
    """
    for _ in range(nt):
        d2_Re = derivee_seconde_num(Re, dx)
        Im = Im + (dt / 2.0) * d2_Re

        d2_Im = derivee_seconde_num(Im, dx)
        Re = Re - (dt / 2.0) * d2_Im
    return Re, Im


if __name__ == "__main__":
    # --- Paramètres : identiques à ceux de simulation_schrodinger.py ---
    nx = 1000
    x, dx = np.linspace(-50.0, 50.0, nx, retstep=True)
    dt = 0.001

    k0 = 2.8
    a = 2.0
    xc = -20.0

    psi_init = generer_paquet_onde(x, xc, k0, a)
    Re, Im = np.real(psi_init), np.imag(psi_init)

    # Instants de comparaison (en nombre de pas de temps depuis t=0)
    instants_nt = [0, 2000, 5000, 8000]

    fig, axes = plt.subplots(1, len(instants_nt), figsize=(5 * len(instants_nt), 4.5), sharey=True)

    nt_precedent = 0
    for ax, nt_cible in zip(axes, instants_nt):
        # On ne propage que les pas supplémentaires nécessaires pour
        # atteindre l'instant cible (on repart de l'état précédent)
        Re, Im = propager_libre(Re, Im, dx, dt, nt_cible - nt_precedent)
        nt_precedent = nt_cible
        t_actuel = nt_cible * dt

        densite_num = Re**2 + Im**2

        # Solution analytique correspondante : GaussWP est centrée en x=0 à
        # t=0, on la décale donc de xc. Le facteur de phase exp(i*k0*xc) est
        # nécessaire pour que la porteuse coïncide exactement (sinon décalage
        # de phase global sans incidence sur |psi|^2, mais on le garde pour
        # comparer aussi Re et Im directement).
        psi_analytique = np.exp(1j * k0 * xc) * GaussWP(k0, a, x - xc, t_actuel)
        densite_analytique = np.abs(psi_analytique)**2

        erreur = np.max(np.abs(densite_num - densite_analytique))
        print(f"t = {t_actuel:5.2f}  ->  erreur max sur |psi|^2 = {erreur:.3e}")

        ax.plot(x, densite_analytique, color="blue", linewidth=2, label="Analytique (GaussWP)")
        ax.plot(x, densite_num, color="red", linestyle="--", label="Numérique (Verlet)")
        ax.set_title(f"t = {t_actuel:.1f}")
        ax.set_xlabel("Position x")
        ax.grid(True)

    axes[0].set_ylabel("Densité |Ψ(x,t)|²")
    axes[0].legend(fontsize="small")

    plt.suptitle("Validation de l'algorithme : particule libre (V=0) vs solution analytique")
    plt.tight_layout()
    plt.savefig("validation_particule_libre.png")
