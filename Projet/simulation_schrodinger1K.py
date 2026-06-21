import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")

from algo_derivation1K import derivee_seconde_num

def generer_barriere_potentiel(x: np.ndarray, debut: float, largeur: float, V0: float) -> np.ndarray:
    """
    Construit un potentiel rectangulaire (V0 sur [debut, debut+largeur], 0 ailleurs).
    """
    V = np.zeros_like(x)
    fin = debut + largeur
    masque_puits = (x >= debut) & (x <= fin)
    V[masque_puits] = V0
    return V

def generer_paquet_onde(x: np.ndarray, xc: float, k0: float, a: float) -> np.ndarray:
    """
    Paquet d'ondes gaussien initial (t = 0).
    """
    prefacteur = (2.0 / (np.pi * a**2)) ** 0.25
    enveloppe = np.exp(-((x - xc)**2) / (a**2))
    porteuse = np.exp(1j * k0 * x)
    return prefacteur * enveloppe * porteuse

def calculer_transmission(x: np.ndarray, dx: float, densite_finale: np.ndarray, psi_initial: np.ndarray, fin_puits: float) -> float:
    """
    Estime le coefficient de transmission T en comparant la probabilité après la barrière
    à l'intégrale de la densité initiale.
    """
    densite_initiale_totale = np.sum(np.abs(psi_initial)**2) * dx
    masque_transmission = x > (fin_puits + 2.0)
    densite_transmise = np.sum(densite_finale[masque_transmission]) * dx
    return densite_transmise / densite_initiale_totale

def simuler_et_animer(x: np.ndarray, dx: float, dt: float, nt: int, V: np.ndarray, psi_initial: np.ndarray) -> np.ndarray:
    """
    Simulation avec stockage 2D (Re_2D, Im_2D) et animation basique (optionnel).
    Renvoie la densité finale (dernier pas de temps).
    """
    nx = len(x)
    Re_2D = np.zeros((nx, nt))
    Im_2D = np.zeros((nx, nt))

    Re_2D[:, 0] = np.real(psi_initial)
    Im_2D[:, 0] = np.imag(psi_initial)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    densite_initiale = Re_2D[:, 0]**2 + Im_2D[:, 0]**2
    ligne_densite, = ax.plot(x, densite_initiale, label="Densité |Ψ|²", color="blue", linewidth=2)
    ligne_potentiel, = ax.plot(x, V / 5.0, label="Potentiel V(x) (échelle)", color="orange", linestyle="--")
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(-1.2, 1.5)
    ax.set_title("Propagation (stockage 2D)"); ax.set_xlabel("x"); ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right"); ax.grid(True)

    for n in range(nt - 1):
        d2_Re = derivee_seconde_num(Re_2D[:, n], dx)
        Im_2D[:, n+1] = Im_2D[:, n] + (dt / 2.0) * d2_Re - dt * V * Re_2D[:, n]
        d2_Im = derivee_seconde_num(Im_2D[:, n+1], dx)
        Re_2D[:, n+1] = Re_2D[:, n] - (dt / 2.0) * d2_Im + dt * V * Im_2D[:, n+1]

        if n % 200 == 0:
            densite_actuelle = Re_2D[:, n+1]**2 + Im_2D[:, n+1]**2
            ligne_densite.set_ydata(densite_actuelle)
            plt.pause(0.001)

    plt.ioff()
    densite_finale = Re_2D[:, -1]**2 + Im_2D[:, -1]**2
    return densite_finale

if __name__ == "__main__":
    nx = 1000
    x_min, x_max = -50.0, 50.0
    x, dx = np.linspace(x_min, x_max, nx, retstep=True)

    nt = 20000
    dt = 0.001
    t, dt_verif = np.linspace(0.0, (nt - 1) * dt, nt, retstep=True)
    assert np.isclose(dt_verif, dt), "Incohérence entre le tableau t et dt"

    V0 = 5.0
    debut_barriere = 5.0
    largeur_a = 0.8

    k0 = 2.8
    a = 2.0
    xc = -20.0

    V = generer_barriere_potentiel(x, debut_barriere, largeur_a, V0)
    psi_init = generer_paquet_onde(x, xc, k0, a)

    densite_fin = simuler_et_animer(x, dx, dt, nt, V, psi_init)
    fin_barriere = debut_barriere + largeur_a
    T = calculer_transmission(x, dx, densite_fin, psi_init, fin_barriere)

    print("\n" + "="*30)
    print(f"Coefficient de transmission T = {T:.4f}")
    print(f"Temps total simulé : t = {t[-1]:.2f} (sur {len(t)} pas de temps)")
    print("="*30 + "\n")

    plt.show()
