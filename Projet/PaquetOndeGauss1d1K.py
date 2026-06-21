####------PaquetOndeGauss1d-----####
# Partie 2 de l'énoncé : Paquet d'ondes gaussien
# Section 2.2 du projet numérique
##

import numpy
from numpy import pi, exp, sqrt, real, imag, linspace
import matplotlib.pyplot as plt
import scipy.integrate

# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 2.2a — Constantes physiques
# ════════════════════════════════════════════════════════════════════════════════

# Constantes avec unités physiques (SI)
HBAR_SI = 1.0546e-34   # constante de Planck réduite [J·s = kg·m²·s⁻¹]
MASS_SI  = 9.109e-31   # masse de l'électron [kg]

# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 2.2b — Fonction GaussWP
# ════════════════════════════════════════════════════════════════════════════════

def GaussWP(k0, a, x, t, hbar=HBAR_SI, m=MASS_SI):
    """
    Calcule le paquet d'ondes gaussien Ψ(x, t) selon la formule (5) de l'énoncé :
    
        Ψ(x, t) = (1/8π³)^(1/4) × √(4πma / (ma² + 2iℏt))
                  × exp( m(a²k₀ + 2ix)² / (4(ma² + 2iℏt)) − a²k₀²/4 )
    
    Origine : c'est le résultat de l'intégrale de la transformée de Fourier du
    paquet gaussien g(k) = √a·(2π)^(-1/4)·exp(-a²(k-k₀)²/4)
    
    Signification des paramètres :
      k0  : nombre d'onde central [rad/m] → l'impulsion moyenne est p₀ = ℏk₀
      a   : largeur réelle du paquet [m] → le paquet est localisé sur ~a en x
      x   : tableau des positions [m]
      t   : instant [s]
      hbar: constante de Planck réduite [J·s]
      m   : masse de la particule [kg]
    
    Paramètres optionnels hbar et m pour pouvoir aussi utiliser des unités naturelles.
    """
    # Dénominateur complexe qui apparaît dans plusieurs termes
    # Ce terme encode à la fois la localisation spatiale (ma²) et l'évolution temporelle (2iℏt)
    denom = m * a**2 + 2j * hbar * t          # [kg·m²] = [J·s²]
    
    # ── Préfacteur de normalisation ──
    # (1/8π³)^(1/4) × √(4πma/denom)
    # Ce préfacteur assure que ∫|Ψ|²dx = 1 à tout instant t
    prefactor = (1.0 / (8.0 * pi**3))**0.25 * sqrt(4.0 * pi * m * a / denom)
    
    # ── Argument de l'exponentielle principale ──
    # Terme 1 : m(a²k₀ + 2ix)² / (4·denom)
    #   → contient l'oscillation exp(ik₀x) et la forme gaussienne
    # Terme 2 : -a²k₀²/4
    #   → terme correctif de normalisation de g(k)
    arg_exp = m * (a**2 * k0 + 2j * x)**2 / (4.0 * denom) - a**2 * k0**2 / 4.0
    
    return prefactor * exp(arg_exp)


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 2.2c — Test et affichage à t = 0
# ════════════════════════════════════════════════════════════════════════════════

def Check_normalization(psi, x):
    """
    Vérifie que ∫|Ψ(x,t)|²dx = 1 (condition de normalisation).
    Retourne la valeur de l'intégrale (doit être ≈ 1.0).
    """
    densite = numpy.abs(psi)**2
    return scipy.integrate.simpson(densite, x=x)


def Plot_paquet_physique():
    """
    Section 2.2c : représente Re(Ψ) et Im(Ψ) à t=0 avec les unités physiques SI.
    
    Illustre AUSSI la difficulté (question 2.2d) : les oscillations sont beaucoup
    trop rapides pour être visibles sur un graphique standard.
    """
    print("=== Version avec unités physiques (SI) ===")
    
    # Paramètres typiques pour un électron libre
    k0_SI = 1.0e10   # nombre d'onde [rad/m], soit λ = 2π/k₀ ≈ 6.3 Å
    a_SI  = 1.0e-9   # largeur du paquet ~ 1 nm (10 Angström)
    t0    = 0.0      # instant initial
    
    # Longueur d'onde λ = 2π/k₀
    lam_SI = 2 * pi / k0_SI
    print(f"  Longueur d'onde : λ = {lam_SI:.2e} m = {lam_SI*1e10:.2f} Å")
    print(f"  Largeur du paquet : a = {a_SI:.2e} m = {a_SI*1e10:.2f} Å")
    
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  DIFFICULTÉ 1 : les échelles sont très différentes                   │
    # │  La longueur d'onde λ ≈ 6.3e-10 m, la largeur a ≈ 1e-9 m           │
    # │  → il faut choisir un intervalle x adapté (~quelques a)             │
    # │  ET suffisamment de points pour résoudre les oscillations de λ      │
    # └─────────────────────────────────────────────────────────────────────┘
    
    # Grille spatiale sur 5 largeurs de paquet, centrée sur 0
    x_SI = linspace(-5 * a_SI, 5 * a_SI, 5000)
    
    psi_SI = GaussWP(k0=k0_SI, a=a_SI, x=x_SI, t=t0, hbar=HBAR_SI, m=MASS_SI)
    
    # Vérification de la normalisation
    norme = Check_normalization(psi_SI, x_SI)
    print(f"  Norme = {norme:.6f}  (devrait être ≈ 1.0)")
    
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  DIFFICULTÉ 2 : les oscillations de Re(Ψ) sont très rapides        │
    # │  Le vrai problème : k₀ = 1e10 rad/m → oscillations sur ~6 Å        │
    # │  Mais la gaussienne s'étend sur ~10 nm → ratio 167 oscillations!   │
    # │  Pour voir à la fois l'enveloppe ET les oscillations, il faudrait  │
    # │  des milliers de points → très lourd en mémoire                    │
    # └─────────────────────────────────────────────────────────────────────┘
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle("Paquet d'ondes gaussien (unités SI) — t = 0", fontsize=12)
    
    x_angstrom = x_SI * 1e10  # conversion m → Angström pour l'affichage
    
    axes[0].plot(x_angstrom, real(psi_SI), color='steelblue', lw=0.8, label=r"Re($\Psi$)")
    axes[0].set_ylabel(r"Re($\Psi$) [m$^{-1/2}$]", fontsize=9)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='gray', lw=0.5)
    
    axes[1].plot(x_angstrom, imag(psi_SI), color='firebrick', lw=0.8, label=r"Im($\Psi$)")
    axes[1].set_ylabel(r"Im($\Psi$) [m$^{-1/2}$]", fontsize=9)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='gray', lw=0.5)
    
    axes[2].plot(x_angstrom, numpy.abs(psi_SI)**2, color='darkviolet', lw=1.5, label=r"$|\Psi|^2$")
    axes[2].set_ylabel(r"$|\Psi|^2$ [m$^{-1}$]", fontsize=9)
    axes[2].set_xlabel("x (Å)", fontsize=11)
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 2.2d+e — Difficulté et solution : unités naturelles
# ════════════════════════════════════════════════════════════════════════════════

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  QUESTION 2.2d : Quelle difficulté rencontrez-vous ?                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DIFFICULTÉ 1 - Amplitudes minuscules :                                      ║
║  La norme impose ∫|Ψ|²dx = 1 avec x en mètres → |Ψ| ~ 1/√m ~ 3×10⁴ m⁻¹/²  ║
║  Ces valeurs très grandes ou très petites causent des problèmes numériques   ║
║                                                                              ║
║  DIFFICULTÉ 2 - Oscillations trop rapides :                                  ║
║  Avec k₀ = 10¹⁰ rad/m, la longueur d'onde est λ = 6.3 Å.                   ║
║  Pour résoudre les oscillations, il faut Δx < λ/10 ≈ 6×10⁻¹¹ m.           ║
║  Si le paquet s'étend sur 10 nm, il faut 10 nm / 6×10⁻¹¹ m ≈ 167 points    ║
║  PAR oscillation, soit des dizaines de milliers de points en tout.          ║
║  → La simulation devient lente et consomme beaucoup de mémoire.             ║
║                                                                              ║
║  DIFFICULTÉ 3 - Temps caractéristiques minuscules :                          ║
║  Le temps de traversée τ ~ a/v_g ≈ 10⁻⁹/(10⁶) ≈ 10⁻¹⁵ s (femtosecondes) ║
║  → Il faut des pas de temps extrêmement petits.                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  QUESTION 2.2e : Solution/astuce                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SOLUTION : Travailler en UNITÉS NATURELLES (ℏ = m = 1)                     ║
║                                                                              ║
║  On choisit des unités telles que ℏ = 1 et m = 1.                           ║
║  Alors tous les nombres sont ~1, faciles à manipuler numériquement.          ║
║  Le code de M. Akridas (PaquetOndes.py) utilise précisément cette approche. ║
║                                                                              ║
║  Physiquement : on travaille avec des variables adimensionnées.              ║
║  x en unités de a (largeur du paquet), t en unités de ma²/ℏ (temps tunnel)  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Constantes en unités naturelles (ℏ = m = 1) ──────────────────────────────
HBAR = 1.0   # constante de Planck réduite (sans unité)
MASS = 1.0   # masse de la particule (sans unité)


def GaussWP_naturel(k0, a, x, t):
    """
    Même formule que GaussWP, mais avec ℏ = m = 1 (unités naturelles).
    
    Beaucoup plus simple numériquement :
      - Tous les nombres sont de l'ordre de 1
      - On peut choisir k₀, a, et t librement sans problème de précision
    
    C'est l'approche utilisée dans PaquetOndes.py du professeur.
    """
    # Note : avec m=1, hbar=1, la formule se simplifie considérablement
    # denom = a² + 2i·t   (au lieu de ma² + 2iℏt)
    denom = a**2 + 2j * t
    
    prefactor = (1.0 / (8.0 * pi**3))**0.25 * sqrt(4.0 * pi * a / denom)
    
    arg_exp = (a**2 * k0 + 2j * x)**2 / (4.0 * denom) - a**2 * k0**2 / 4.0
    
    return prefactor * exp(arg_exp)


def Plot_paquet_naturel():
    """
    Section 2.2e : affichage avec unités naturelles → résout toutes les difficultés.
    Montre aussi l'évolution du paquet au cours du temps.
    """
    print("\n=== Version avec unités naturelles (ℏ = m = 1) ===")
    
    # Paramètres
    k0 = 2.0      # nombre d'onde central (sans unité)
    a  = 2.0      # largeur du paquet dans l'espace réel (sans unité)
    
    # Grille spatiale simple
    x = linspace(-15, 30, 2000)
    
    # Quelques instants pour montrer l'évolution
    temps = [0.0, 1.0, 3.0, 5.0]
    
    fig, axes = plt.subplots(2, len(temps), figsize=(14, 7))
    fig.suptitle("Paquet d'ondes gaussien — Unités naturelles (ℏ = m = 1)", fontsize=12)
    
    for i, t in enumerate(temps):
        psi = GaussWP_naturel(k0=k0, a=a, x=x, t=t)
        
        # Vérification de la normalisation à chaque instant
        norme = Check_normalization(psi, x)
        
        # Ligne du haut : densité de probabilité |Ψ|²
        axes[0, i].plot(x, numpy.abs(psi)**2, color='darkviolet', lw=1.8)
        axes[0, i].set_title(f"t = {t}  |  norme = {norme:.3f}", fontsize=9)
        axes[0, i].set_ylabel(r"$|\Psi|^2$" if i == 0 else "", fontsize=9)
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_ylim(bottom=0)
        
        # Ligne du bas : partie réelle et imaginaire
        axes[1, i].plot(x, real(psi), color='steelblue', lw=1.2, label=r"Re($\Psi$)")
        axes[1, i].plot(x, imag(psi), color='firebrick', lw=1.2, label=r"Im($\Psi$)")
        axes[1, i].axhline(0, color='gray', lw=0.5, ls='--')
        axes[1, i].set_xlabel("x", fontsize=9)
        if i == 0:
            axes[1, i].set_ylabel(r"Re/Im($\Psi$)", fontsize=9)
            axes[1, i].legend(fontsize=7)
        axes[1, i].grid(True, alpha=0.3)
        
        print(f"  t = {t:.1f} : norme = {norme:.6f}")
    
    # Vitesse de groupe théorique : v_g = ℏk₀/m = k₀ (avec ℏ=m=1)
    v_g = HBAR * k0 / MASS
    print(f"\n  Vitesse de groupe théorique : v_g = ℏk₀/m = {v_g:.2f} (unités naturelles)")
    print(f"  Position à t=5 : x₀ + v_g·t = 0 + {v_g}×5 = {v_g*5:.1f}")
    
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Partie 2.2 : Paquet d'ondes gaussien ===\n")
    
    # 2.2c : test avec unités physiques → montre la difficulté
    Plot_paquet_physique()
    
    # 2.2e : solution avec unités naturelles
    Plot_paquet_naturel()
