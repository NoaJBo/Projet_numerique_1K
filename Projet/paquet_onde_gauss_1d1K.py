import numpy as np
import matplotlib.pyplot as plt

# Unités réduites (hbar = m = 1) :
# on adopte la même convention que simulation_schrodinger.py / etude_*.py,
# où l'équation codée est i*dPsi/dt = -(1/2)*d2Psi/dx^2 + V*Psi, ce qui
# correspond implicitement à hbar = m = 1 dans l'équation de Schrödinger
# générale i*hbar*dPsi/dt = -(hbar^2/2m)*d2Psi/dx^2 + V*Psi.
# Sans cette unification, comparer GaussWP (avec hbar, m réels de l'électron)
# à la simulation numérique (en hbar=m=1) donnerait des échelles de temps
# totalement incompatibles.
#
# Valeurs réelles de l'électron (pour mémoire / réponses analytiques en SI) :
#   hbar_SI = 1.054571817e-34  J.s
#   m_SI    = 9.1093837015e-31 kg
hbar = 1.0
m = 1.0

def GaussWP(k0: float, a: float, x: np.ndarray, t: float) -> np.ndarray:
    """
    Retourne le paquet d'ondes gaussien psi(x, t)
    """
    if t == 0:
        prefacteur = (2.0 / (np.pi * a**2))**0.25
        enveloppe = np.exp(-(x**2) / (a**2))
        porteuse = np.exp(1j * k0 * x)

        return prefacteur * enveloppe * porteuse
    else:
        # Évolution temporelle du paquet d'ondes gaussien (éq. 5 du sujet) :
        #
        #            (1/8pi^3)^(1/4) * sqrt(4*pi*m*a / (m*a^2 + 2i*hbar*t))
        #            * exp[ (m/4)*(a^2*k0 + 2ix)^2 / (m*a^2 + 2i*hbar*t) - a^2*k0^2/4 ]
        denom = m * a**2 + 2j * hbar * t

        terme1 = (1.0 / (8.0 * np.pi**3))**0.25
        terme2 = np.sqrt(4.0 * np.pi * m * a / denom)
        exposant = (m / 4.0) * (a**2 * k0 + 2j * x)**2 / denom - (a**2 * k0**2) / 4.0

        return terme1 * terme2 * np.exp(exposant)

if __name__ == "__main__":
    k0_val = 10.0
    a_val = 2.0

    x_array = np.linspace(-10, 10, 1000)

    psi_0 = GaussWP(k0_val, a_val, x_array, 0.0)

    # Graph
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_array, np.real(psi_0), label="Partie réelle", color="blue", linewidth=1.5)
    ax.plot(x_array, np.imag(psi_0), label="Partie imaginaire", color="red", linestyle="--", linewidth=1.5)
    densite = np.abs(psi_0)**2
    ax.plot(x_array, densite, label="Densité de probilité |psi|²", color="black", linestyle=":", linewidth=2)

    ax.set_title("paquet d'ondes gaussien à l'instant t = 0")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig("paquet_onde.png")

    # ---------------------------------------------------------
    # 2.2.2.d/e : Difficulté rencontrée et astuce
    # ---------------------------------------------------------
    # Difficulté initiale : avec m et hbar réels (électron), le terme
    # 2*hbar*t/(m*a^2) est négligeable sauf pour des t astronomiques (~1e4 s) :
    # rien ne semble bouger si on prend des t "raisonnables" (1.0, 2.0 s...).
    #
    # Astuce générale (valable quelle que soit la convention d'unités) :
    # m*a^2 + 2i*hbar*t = m*a^2*(1 + i*t/tau) avec tau = m*a^2/(2*hbar).
    # La FORME du paquet ne dépend donc QUE du rapport t/tau, jamais des
    # valeurs réelles de m et hbar. On exprime donc t en multiples de ce
    # temps caractéristique de dispersion tau.
    #
    # Remarque : on travaille maintenant en unités réduites hbar = m = 1
    # (cf. en-tête du fichier), pour être cohérent avec simulation_schrodinger.py
    # et pouvoir comparer directement les deux approches (partie 3.2.5 du sujet).
    # Dans cette convention, tau = a^2/2, une échelle de temps directement
    # comparable à nt*dt utilisé dans les simulations numériques.
    tau = m * a_val**2 / (2.0 * hbar)
    print(f"Temps caractéristique de dispersion : tau = {tau:.3e} (unités réduites)")

    # Le centre du paquet se déplace à la vitesse de groupe v_g = hbar*k0/m.
    # On recentre la fenêtre d'affichage sur ce centre à chaque instant pour
    # isoler visuellement l'étalement (qui ne dépend pas de k0), sans quoi le
    # paquet sort simplement du cadre.
    v_g = hbar * k0_val / m

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for facteur in [0.0, 0.5, 1.0, 2.0, 5.0]:
        t_test = facteur * tau
        centre_t = v_g * t_test
        x_eval = x_array + centre_t  # positions absolues à évaluer

        psi_t = GaussWP(k0_val, a_val, x_eval, t_test)
        densite_t = np.abs(psi_t)**2
        ax2.plot(x_array, densite_t, label=fr"t = {facteur}$\tau$")  # affiché en position relative au centre

    ax2.set_title(r"Étalement du paquet d'ondes gaussien (recentré, en multiples de $\tau$)")
    ax2.set_xlabel("Position relative au centre du paquet")
    ax2.set_ylabel("Densité |Ψ(x,t)|²")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("paquet_onde_dispersion.png")
