#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from algo_derivation1K import derivee_seconde_num
from paquet_onde_gauss_1d1K import GaussWP, hbar, m  # unités réduites hbar = m = 1

def generer_paquet_onde(x: np.ndarray, xc: float, k0: float, a: float) -> np.ndarray:
    """Génère le paquet d'ondes gaussien initial (identique à simulation_schrodinger)."""
    prefacteur = (2.0 / (np.pi * a**2)) ** 0.25
    enveloppe = np.exp(-((x - xc)**2) / (a**2))
    porteuse = np.exp(1j * k0 * x)
    return prefacteur * enveloppe * porteuse

def propager_libre(Re: np.ndarray, Im: np.ndarray, dx: float, dt: float, nt: int):
    """Propage Re/Im nt pas pour particule libre (V = 0) et renvoie Re, Im finaux."""
    for _ in range(nt):
        d2_Re = derivee_seconde_num(Re, dx)
        Im = Im + (dt / 2.0) * d2_Re
        d2_Im = derivee_seconde_num(Im, dx)
        Re = Re - (dt / 2.0) * d2_Im
    return Re, Im

if __name__ == "__main__":
    nx = 1000
    x, dx = np.linspace(-50.0, 50.0, nx, retstep=True)
    dt = 0.001

    k0 = 2.8
    a = 2.0
    xc = -20.0

    psi_init = generer_paquet_onde(x, xc, k0, a)
    Re, Im = np.real(psi_init), np.imag(psi_init)

    instants_nt = [0, 2000, 5000, 8000]
    fig, axes = plt.subplots(1, len(instants_nt), figsize=(5 * len(instants_nt), 4.5), sharey=True)

    nt_precedent = 0
    for ax, nt_cible in zip(axes, instants_nt):
        Re, Im = propager_libre(Re, Im, dx, dt, nt_cible - nt_precedent)
        nt_precedent = nt_cible
        t_actuel = nt_cible * dt
        densite_num = Re**2 + Im**2

        psi_analytique = np.exp(1j * k0 * xc) * GaussWP(k0, a, x - xc, t_actuel)
        densite_analytique = np.abs(psi_analytique)**2

        erreur = np.max(np.abs(densite_num - densite_analytique))
        print(f"t = {t_actuel:5.2f}  ->  erreur max sur |psi|^2 = {erreur:.3e}")

        ax.plot(x, densite_analytique, color="blue", linewidth=2, label="Analytique (GaussWP)")
        ax.plot(x, densite_num, color="red", linestyle="--", label="Numérique (Verlet)")
        ax.set_title(f"t = {t_actuel:.1f}"); ax.set_xlabel("Position x"); ax.grid(True)

    axes[0].set_ylabel("Densité |Ψ(x,t)|²"); axes[0].legend(fontsize="small")
    plt.suptitle("Validation : particule libre vs solution analytique")
    plt.tight_layout(); plt.savefig("validation_particule_libre.png")
