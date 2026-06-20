import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")

from algo_derivation import derivee_seconde_num

def generer_puits_potentiel(x: np.ndarray, debut: float, largeur: float, V0: float) -> np.ndarray:
    """
    Génère et retourne le tableau représentant le puits de potentiel V(x).
    Le potentiel vaut V0 à l'intérieur du puits et 0 à l'extérieur.
    """
    V = np.zeros_like(x)
    fin = debut + largeur
    masque_puits = (x >= debut) & (x <= fin)
    V[masque_puits] = V0
    return V

def generer_paquet_onde(x: np.ndarray, xc: float, k0: float, a: float) -> np.ndarray:
    """
    Génère et retourne le paquet d'ondes gaussien initial complexe à l'instant t=0.
    """
    prefacteur = (2.0 / (np.pi * a**2)) ** 0.25
    enveloppe = np.exp(-((x - xc)**2) / (a**2))
    porteuse = np.exp(1j * k0 * x)
    return prefacteur * enveloppe * porteuse

def calculer_transmission(x: np.ndarray, dx: float, densite_finale: np.ndarray, psi_initial: np.ndarray, fin_puits: float) -> float:
    """
    Calcule le coefficient de transmission T.
    Fait le rapport entre l'intégrale de la probabilité de présence APRES le puits
    à la fin de la simulation, et l'intégrale de l'onde initiale.
    """
    # On calcule l'intégrale de la probabilité initiale (devrait valoir ~1 si bien normalisé)
    densite_initiale_totale = np.sum(np.abs(psi_initial)**2) * dx
    
    # On isole la zone située juste après le puits
    masque_transmission = x > (fin_puits + 2.0)
    densite_transmise = np.sum(densite_finale[masque_transmission]) * dx
    
    return densite_transmise / densite_initiale_totale

def simuler_et_animer(x: np.ndarray, dx: float, dt: float, nt: int, V: np.ndarray, psi_initial: np.ndarray) -> np.ndarray:
    """
    Exécute l'algorithme de Verlet pour l'équation de Schrödinger.
    Gère l'affichage en temps réel et retourne le tableau de densité de probabilité final.
    """
    Re = np.real(psi_initial)
    Im = np.imag(psi_initial)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    densite = Re**2 + Im**2
    ligne_densite, = ax.plot(x, densite, label="Densité de probabilité |Psi|²", color="blue", linewidth=2)
    # On divise V par 10 uniquement pour que l'affichage soit à la même échelle que l'onde
    ligne_potentiel, = ax.plot(x, V / 10.0, label="Potentiel V(x) (échelle 1:10)", color="orange", linestyle="--")

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-1.2, 1.5)
    ax.set_title("Effet Ramsauer-Townsend : Simulation de Schrödinger")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    ax.grid(True)

    print("Début de la simulation...")
    
    for n in range(nt):
        d2_Re = derivee_seconde_num(Re, dx)
        Im = Im + (dt / 2.0) * d2_Re - dt * V * Re

        d2_Im = derivee_seconde_num(Im, dx)
        Re = Re - (dt / 2.0) * d2_Im + dt * V * Im

        if n % 50 == 0:
            densite_actuelle = Re**2 + Im**2
            ligne_densite.set_ydata(densite_actuelle)
            plt.pause(0.001)

    print("Simulation terminée !")
    plt.ioff()
    
    densite_finale = Re**2 + Im**2
    return densite_finale


if __name__ == "__main__":
    nx = 1000
    x_min, x_max = -50.0, 50.0
    x, dx = np.linspace(x_min, x_max, nx, retstep=True)

    nt = 2000
    dt = 0.01

    # Paramètres du puits de potentiel
    V0 = -10.0
    debut_puits = 5.0
    largeur_a = 5.0

    # Paramètres du paquet d'ondes initial
    k0 = 2.0
    a = 2.0
    xc = -20.0

    V = generer_puits_potentiel(x, debut_puits, largeur_a, V0)
    psi_init = generer_paquet_onde(x, xc, k0, a)

    densite_fin = simuler_et_animer(x, dx, dt, nt, V, psi_init)

    fin_puits = debut_puits + largeur_a
    T = calculer_transmission(x, dx, densite_fin, psi_init, fin_puits)

    print("\n" + "="*30)
    print(f"Coefficient de transmission T = {T:.4f}")
    print("="*30 + "\n")

    plt.show()
