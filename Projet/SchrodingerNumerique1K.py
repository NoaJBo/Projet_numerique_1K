####------SchrodingerNumerique-----####
# Partie 3 de l'énoncé : Résolution numérique de l'équation de Schrödinger
# Sections 3.1 (algorithme de dérivation) et 3.2 (algorithme pour Schrödinger)
# Unités naturelles : ℏ = m = 1
##

import numpy
from numpy import pi, exp, sqrt
import matplotlib.pyplot as plt
import scipy.integrate

# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 3.1 — Algorithme de dérivation numérique
# ════════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────────
# 3.1.1 — DÉRIVÉE PREMIÈRE
# Rappel mathématique :
#   df/dx = lim_{h→0} [f(x+h) - f(x)] / h
#
# Numériquement, on ne peut pas prendre h→0 exactement.
# On choisit h = dx (le pas de la grille) et on approche :
#
#   Différence progressive (forward) :   df/dx ≈ [f(x+h) - f(x)] / h
#   Différence centrée (central) :        df/dx ≈ [f(x+h) - f(x-h)] / (2h)
#
# La différence centrée est plus précise (erreur O(h²) vs O(h) pour progressive).
# ────────────────────────────────────────────────────────────────────────────────

def Derivee_premiere_progressive(f_array, dx):
    """
    Calcule la dérivée première df/dx par différences progressives.
    
    Méthode : [f(x+h) - f(x)] / h
    Précision : ordre 1 en h (erreur ~ h/2 × f'')
    
    Paramètre:
      f_array : tableau numpy 1D contenant les valeurs de f
      dx      : pas d'espace (distance entre deux points consécutifs)
    
    Retourne:
      Tableau de même taille que f_array.
      Remarque : le DERNIER point ne peut pas être calculé 
                 (il faudrait f[n+1] qui n'existe pas)
                 → on copie la valeur précédente pour garder la même taille.
    """
    n = len(f_array)
    df = numpy.zeros(n)  # initialise le tableau résultat à zéro
    
    # Boucle sur tous les points sauf le dernier
    for i in range(n - 1):
        # Différence progressive : utilise le point à droite
        df[i] = (f_array[i + 1] - f_array[i]) / dx
    
    # Dernier point : on ne peut pas aller plus loin → on répète la valeur précédente
    df[-1] = df[-2]
    
    return df


def Derivee_premiere_centree(f_array, dx):
    """
    Calcule la dérivée première df/dx par différences centrées.
    
    Méthode : [f(x+h) - f(x-h)] / (2h)
    Précision : ordre 2 en h (erreur ~ h²/6 × f''')
    → DEUX FOIS plus précise que la méthode progressive à même pas h.
    
    Paramètre:
      f_array : tableau numpy 1D contenant les valeurs de f
      dx      : pas d'espace
    
    Retourne:
      Tableau de même taille que f_array.
      Remarque : les bords (premier et dernier point) sont traités avec
                 la méthode progressive pour éviter les accès hors tableau.
    """
    n = len(f_array)
    df = numpy.zeros(n)
    
    # Points intérieurs : différence centrée
    for i in range(1, n - 1):
        df[i] = (f_array[i + 1] - f_array[i - 1]) / (2 * dx)
    
    # Bords : différence progressive (on n'a pas de voisin d'un côté)
    df[0]  = (f_array[1]  - f_array[0])   / dx   # bord gauche
    df[-1] = (f_array[-1] - f_array[-2])  / dx   # bord droit
    
    return df


def Derivee_premiere_centree_numpy(f_array, dx):
    """
    VERSION VECTORISÉE (sans boucle) de la dérivée centrée.
    
    Beaucoup plus rapide grâce aux opérations tableau numpy.
    C'est la forme qu'on utilise dans la pratique pour la simulation.
    
    numpy.zeros_like crée un tableau de zéros de même forme que f_array.
    f_array[2:] = tous les éléments sauf les deux premiers
    f_array[:-2] = tous les éléments sauf les deux derniers
    → df[1:-1] = (f[2:] - f[:-2]) / (2*dx)  en une seule opération !
    """
    df = numpy.zeros_like(f_array, dtype=float)
    df[1:-1] = (f_array[2:] - f_array[:-2]) / (2 * dx)
    # Bords
    df[0]  = (f_array[1]  - f_array[0])  / dx
    df[-1] = (f_array[-1] - f_array[-2]) / dx
    return df


# ────────────────────────────────────────────────────────────────────────────────
# 3.1.1c — Fonctions x² et 2x pour le test
# ────────────────────────────────────────────────────────────────────────────────

def carre(x):
    """Retourne x². La dérivée analytique est 2x."""
    return x**2

def deux_x(x):
    """Retourne 2x. C'est la dérivée analytique de x²."""
    return 2 * x


def Test_derivee_premiere():
    """
    Section 3.1.1d : vérifie que la dérivée numérique de x² donne bien 2x.
    Montre l'erreur relative commise.
    """
    print("=== Test : dérivée numérique de f(x) = x² (doit donner 2x) ===\n")
    
    # Grille de test
    x = numpy.linspace(0.5, 5.0, 200)   # on évite x=0 pour l'erreur relative
    dx = x[1] - x[0]
    
    # Valeurs de la fonction et sa dérivée analytique
    f_vals         = carre(x)      # f(x) = x²
    df_analytique  = deux_x(x)     # f'(x) = 2x  (exact)
    
    # Dérivées numériques
    df_progressive = Derivee_premiere_progressive(f_vals, dx)
    df_centree     = Derivee_premiere_centree(f_vals, dx)
    
    # Erreur relative : |numérique - analytique| / |analytique|
    # On ignore les bords (là où la méthode progressive est moins précise)
    idx = slice(10, -10)   # on compare sur les points intérieurs seulement
    err_progressive = numpy.abs(df_progressive[idx] - df_analytique[idx]) / numpy.abs(df_analytique[idx])
    err_centree     = numpy.abs(df_centree[idx]     - df_analytique[idx]) / numpy.abs(df_analytique[idx])
    
    print(f"  Pas dx = {dx:.4f}")
    print(f"  Erreur relative max (méthode progressive) : {numpy.max(err_progressive):.2e}")
    print(f"  Erreur relative max (méthode centrée)     : {numpy.max(err_centree):.2e}")
    print(f"  → La méthode centrée est ~{numpy.max(err_progressive)/numpy.max(err_centree):.0f}x plus précise\n")
    
    # Graphique de comparaison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(x, df_analytique, 'k-', lw=2.5, label="Analytique : 2x")
    ax1.plot(x, df_progressive, 'b--', lw=1.5, label="Numérique (progressive)")
    ax1.plot(x, df_centree, 'r:', lw=1.5, label="Numérique (centrée)")
    ax1.set_title("Dérivée numérique de x²", fontsize=11)
    ax1.set_xlabel("x"); ax1.set_ylabel("df/dx")
    ax1.legend(); ax1.grid(True, alpha=0.4)
    
    ax2.semilogy(x[idx], err_progressive, 'b-', label="Erreur progressive")
    ax2.semilogy(x[idx], err_centree, 'r-', label="Erreur centrée")
    ax2.set_title("Erreur relative |numérique - analytique|/|analytique|", fontsize=10)
    ax2.set_xlabel("x"); ax2.set_ylabel("Erreur relative (échelle log)")
    ax2.legend(); ax2.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────────────────────
# 3.1.2 — DÉRIVÉE SECONDE
# Rappel mathématique :
#   d²f/dx² ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
#
# C'est la formule des DIFFÉRENCES FINIES CENTRÉES pour la dérivée seconde.
# Elle est dérivée en combinant deux différences progressives :
#   f'(x+h/2) ≈ [f(x+h) - f(x)] / h     (dérivée à droite)
#   f'(x-h/2) ≈ [f(x) - f(x-h)] / h     (dérivée à gauche)
#   f''(x) ≈ [f'(x+h/2) - f'(x-h/2)] / h = [f(x+h) - 2f(x) + f(x-h)] / h²
# ────────────────────────────────────────────────────────────────────────────────

def Derivee_seconde(f_array, dx):
    """
    Calcule la dérivée seconde d²f/dx² par différences finies centrées.
    
    Formule : [f(x+h) - 2f(x) + f(x-h)] / h²
    Précision : ordre 2 en h (erreur ~ h²/12 × f⁴)
    
    C'est exactement ce terme qui apparaît dans l'équation de Schrödinger :
      iℏ ∂ψ/∂t = -(ℏ²/2m) × (∂²ψ/∂x²) + V(x)ψ
                              ↑
                   Ce terme qu'on calcule ici !
    """
    n = len(f_array)
    d2f = numpy.zeros_like(f_array)
    
    # Version vectorisée (sans boucle) : efficace et lisible
    # f_array[2:]   = f(x + dx) pour tous les points intérieurs
    # f_array[1:-1] = f(x) pour tous les points intérieurs
    # f_array[:-2]  = f(x - dx) pour tous les points intérieurs
    d2f[1:-1] = (f_array[2:] - 2.0 * f_array[1:-1] + f_array[:-2]) / dx**2
    
    # Bords : on met 0 (condition aux bords absorbantes, traitée dans Schrödinger)
    d2f[0]  = 0.0
    d2f[-1] = 0.0
    
    return d2f


def Test_derivee_seconde():
    """
    Section 3.1.2 : vérifie la dérivée seconde sur un exemple connu.
    Utilise sin(x) → la dérivée seconde est -sin(x).
    """
    print("=== Test : dérivée seconde de f(x) = sin(x) (doit donner -sin(x)) ===\n")
    
    x  = numpy.linspace(0, 4 * pi, 500)
    dx = x[1] - x[0]
    
    f_vals       = numpy.sin(x)
    d2f_analytique = -numpy.sin(x)   # analytique : d²(sin)/dx² = -sin
    d2f_numerique  = Derivee_seconde(f_vals, dx)
    
    # Erreur (on ignore les quelques points de bord)
    idx = slice(5, -5)
    erreur = numpy.max(numpy.abs(d2f_numerique[idx] - d2f_analytique[idx]))
    print(f"  Pas dx = {dx:.4f}")
    print(f"  Erreur absolue max : {erreur:.2e}")
    print(f"  Erreur relative max : {erreur/1.0:.2e}  (divisée par max(|-sin(x)|)=1)\n")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, d2f_analytique, 'k-', lw=2.5, label=r"Analytique : $-\sin(x)$")
    ax.plot(x, d2f_numerique, 'r--', lw=1.5, label="Numérique")
    ax.set_title("Dérivée seconde numérique de sin(x)", fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel(r"$d^2f/dx^2$")
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 3.2 — Algorithme pour l'équation de Schrödinger
# ════════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────────
# L'équation de Schrödinger à 1D pour une particule dans un potentiel V₀ :
#
#   iℏ ∂ψ/∂t = -(ℏ²/2m) ∂²ψ/∂x² + V₀ψ
#
# En unités naturelles (ℏ=m=1) :
#   i ∂ψ/∂t = -(1/2) ∂²ψ/∂x² + V₀ψ
#
# MÉTHODE NUMÉRIQUE — Séparation Réelle/Imaginaire :
# On pose ψ = R + iI où R = Re(ψ) et I = Im(ψ).
# En substituant dans l'équation de Schrödinger, on obtient deux équations couplées :
#
#   ∂I/∂t = -(1/2)·∂²R/∂x² + V₀·R   ... (1)
#   ∂R/∂t =  (1/2)·∂²I/∂x² - V₀·I   ... (2)
#
# En appliquant des différences finies (dérivée temporelle progressive, spatiale centrée) :
#
#   I^{j+1}_i = I^j_i - s·(R^j_{i+1} + R^j_{i-1} - 2R^j_i) + Δt·V_i·R^j_i   ... (1) discrétisée
#   R^{j+1}_i = R^j_i + s·(I^{j+1}_{i+1} + I^{j+1}_{i-1} - 2I^{j+1}_i) - Δt·V_i·I^{j+1}_i  ... (2) discrétisée
#
# où s = Δt/(2·Δx²)  (avec ℏ=m=1)
#
# ASTUCE SEMI-IMPLICITE : on met à jour I en premier, puis on utilise le NOUVEAU I
# pour calculer R. Cela améliore la stabilité numérique.
#
# CONDITION DE STABILITÉ : s = Δt/(2·Δx²) < 0.25  ←←←  À RESPECTER ABSOLUMENT
# Si s > 0.25, les erreurs numériques explosent (la norme diverge).
# ────────────────────────────────────────────────────────────────────────────────

# Paramètres de la simulation (unités naturelles ℏ = m = 1)
HBAR = 1.0
MASS = 1.0

# Paramètres du paquet d'ondes initial
K0    = 2.0    # nombre d'onde central [rad]
A_WP  = 2.0    # largeur du paquet dans l'espace réel (paramètre a de l'énoncé)
X0    = -10.0  # position initiale du centre du paquet

# Grille spatiale
X_MIN, X_MAX = -20.0, 40.0
N_X          = 800             # nombre de points spatiaux
DX           = (X_MAX - X_MIN) / (N_X - 1)
x            = numpy.linspace(X_MIN, X_MAX, N_X)

# Grille temporelle
T_MAX  = 10.0
DT     = 0.0005               # pas de temps
N_T    = int(T_MAX / DT)      # nombre de pas de temps

# Coefficient de stabilité : doit être < 0.25
S = HBAR * DT / (2.0 * MASS * DX**2)
print(f"\nParamètres numériques :")
print(f"  DX = {DX:.5f},  DT = {DT},  s = {S:.5f}")
if S < 0.25:
    print(f"  ✓ Condition de stabilité respectée (s = {S:.4f} < 0.25)")
else:
    print(f"  ✗ ATTENTION : s = {S:.4f} > 0.25 → instabilité probable !")
    print(f"    Réduire DT ou augmenter DX.")


# ────────────────────────────────────────────────────────────────────────────────
# 3.2.2 — Paquet d'ondes gaussien initial
# ────────────────────────────────────────────────────────────────────────────────

def Compute_initial_wavepacket(x, k0=K0, a=A_WP, x0=X0):
    """
    Calcule le paquet d'ondes gaussien initial Ψ(x, t=0).
    
    Utilise la formule normalisée avec ℏ = m = 1 (unités naturelles) :
      Ψ(x, 0) ∝ exp(-(x-x₀)²·a²/2) × exp(ik₀(x-x₀))
    
    Signification :
      - exp(-(x-x₀)²·a²/2) : gaussienne centrée sur x₀ (enveloppe)
      - exp(ik₀(x-x₀))     : onde porteuse (oscillation rapide)
    
    Note : on centre sur x₀ pour placer le paquet à gauche de la barrière.
    """
    # Facteur de normalisation : (a/√π)^(1/2)
    norme = (a / numpy.sqrt(pi))**0.5
    
    # Enveloppe gaussienne × onde porteuse
    gaussienne = numpy.exp(-a**2 * (x - x0)**2 / 2.0)
    porteuse   = numpy.exp(1j * k0 * (x - x0))
    
    return norme * gaussienne * porteuse


# ────────────────────────────────────────────────────────────────────────────────
# 3.2.2 — Tableau 2D de la fonction d'onde (comme demandé dans l'énoncé)
# ────────────────────────────────────────────────────────────────────────────────

def Initialize_psi_2D(nx=N_X, nt=N_T):
    """
    Crée le tableau 2D de la fonction d'onde.
    
    L'énoncé demande :
      "Définir une fonction d'onde (tableau 2d) contenant nx lignes et nt colonnes.
       La première ligne doit contenir un paquet d'ondes gaussien à instant donné
       et le reste du tableau doit contenir des zéros (ou mieux des nombres
       aléatoires empty)."
    
    Structure choisie : psi_2D[j, i]
      - j = indice temporel (j=0 → t=0, j=1 → t=Δt, ...)
      - i = indice spatial  (i=0 → x=X_MIN, i=1 → x=X_MIN+Δx, ...)
    
    Donc : nt lignes × nx colonnes
    → La première LIGNE (j=0) est le paquet initial en fonction de x.
    
    Note : l'énoncé dit "nx lignes et nt colonnes" mais la première "ligne"
    qui contient le paquet doit parcourir les positions x → on interprète
    "première ligne" comme "première tranche temporelle j=0".
    """
    # numpy.empty alloue la mémoire sans initialiser → plus rapide que zeros
    # On utilise complex car ψ est complexe
    psi_2D = numpy.empty((nt, nx), dtype=complex)
    
    # La première ligne (t=0) = paquet d'ondes gaussien initial
    psi_2D[0, :] = Compute_initial_wavepacket(x)
    
    # Le reste n'est pas encore calculé → sera rempli par l'algorithme
    # (numpy.empty les remplit avec des valeurs aléatoires/non-initialisées)
    # Si on voulait des zéros : psi_2D[1:, :] = 0.0
    
    return psi_2D


# ────────────────────────────────────────────────────────────────────────────────
# 3.2.4 — Algorithme d'évolution (cœur de la simulation)
# ────────────────────────────────────────────────────────────────────────────────

def Evoluer_schrodinger(V=None, save_every=200):
    """
    Fait évoluer la fonction d'onde selon l'équation de Schrödinger.
    
    Algorithme : différences finies, méthode R/I semi-implicite.
    
    Le schéma en pseudo-code :
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Initialisation :                                                     │
    │    R[i] = Re(ψ₀(xᵢ))                                                │
    │    I[i] = Im(ψ₀(xᵢ))                                                │
    │                                                                       │
    │  Pour chaque pas de temps j = 0, 1, 2, ..., N_T-1 :                 │
    │    Pour chaque point intérieur i = 1, ..., N_X-2 :                  │
    │      I[i] ← I[i] - s×(R[i+1] + R[i-1] - 2R[i]) + Δt×V[i]×R[i]    │
    │    Pour chaque point intérieur i = 1, ..., N_X-2 :                  │
    │      R[i] ← R[i] + s×(I[i+1] + I[i-1] - 2I[i]) - Δt×V[i]×I[i]    │
    │    Conditions aux bords : R[0]=R[-1]=I[0]=I[-1]=0                   │
    └──────────────────────────────────────────────────────────────────────┘
    
    Paramètres:
      V          : tableau du potentiel V(x), shape (N_X,). Si None → particule libre.
      save_every : sauvegarde la densité tous les "save_every" pas de temps.
    
    Retourne:
      densites_sauvees  : tableau des densités |ψ(x,t)|² aux instants sauvegardés
      temps_sauvegardes : instants correspondants
    """
    # Potentiel (zéro si particule libre)
    if V is None:
        V = numpy.zeros(N_X)
    
    # Initialisation à partir du paquet gaussien
    psi0 = Compute_initial_wavepacket(x)
    R = numpy.real(psi0).copy()   # partie réelle
    I = numpy.imag(psi0).copy()   # partie imaginaire
    
    # Listes pour stocker les résultats au cours du temps
    densites_sauvees  = []
    temps_sauvegardes = []
    normes_sauvees    = []
    
    # Sauvegarde de l'état initial
    densite_init = R**2 + I**2
    densites_sauvees.append(densite_init.copy())
    temps_sauvegardes.append(0.0)
    normes_sauvees.append(scipy.integrate.simpson(densite_init, x=x))
    
    # ── Boucle temporelle principale ──────────────────────────────────────
    for j in range(N_T):
        
        # ── Étape 1 : mise à jour de la partie IMAGINAIRE I ──
        # En utilisant la partie réelle R actuelle
        # Formule : I^{j+1}_i = I^j_i - s×(R_{i+1} + R_{i-1} - 2R_i) + Δt×V_i×R_i
        # Version vectorisée (beaucoup plus rapide que la boucle) :
        I[1:-1] = (
            I[1:-1]
            - S * (R[2:] + R[:-2] - 2.0 * R[1:-1])   # terme de diffusion (énergie cinétique)
            + DT * V[1:-1] * R[1:-1]                   # terme potentiel
        )
        
        # ── Étape 2 : mise à jour de la partie RÉELLE R ──
        # En utilisant le NOUVEAU I (c'est le semi-implicite)
        # Formule : R^{j+1}_i = R^j_i + s×(I_{i+1} + I_{i-1} - 2I_i) - Δt×V_i×I_i
        R[1:-1] = (
            R[1:-1]
            + S * (I[2:] + I[:-2] - 2.0 * I[1:-1])   # terme de diffusion
            - DT * V[1:-1] * I[1:-1]                   # terme potentiel
        )
        
        # ── Conditions aux bords : ψ = 0 aux extrémités ──
        # Modélise une boîte infinie : la particule ne peut pas sortir.
        # Évite aussi les réflexions parasites sur les bords.
        R[0]  = R[-1]  = 0.0
        I[0]  = I[-1]  = 0.0
        
        # ── Sauvegarde périodique ──
        if j % save_every == 0:
            densite = R**2 + I**2
            densites_sauvees.append(densite.copy())
            temps_sauvegardes.append((j + 1) * DT)
            normes_sauvees.append(scipy.integrate.simpson(densite, x=x))
    
    print(f"\nSimulation terminée :")
    print(f"  Norme initiale : {normes_sauvees[0]:.6f}")
    print(f"  Norme finale   : {normes_sauvees[-1]:.6f}")
    print(f"  Variation      : {abs(normes_sauvees[-1] - normes_sauvees[0]):.2e}")
    print(f"  → {'✓ Norme conservée' if abs(normes_sauvees[-1]-normes_sauvees[0])<0.01 else '✗ Norme NON conservée (ajuster DT)'}")
    
    return (numpy.array(densites_sauvees),
            numpy.array(temps_sauvegardes),
            numpy.array(normes_sauvees))


# ────────────────────────────────────────────────────────────────────────────────
# 3.2.5 — Comparaison avec le paquet d'ondes analytique
# ────────────────────────────────────────────────────────────────────────────────

def GaussWP_analytique(x, t, k0=K0, a=A_WP, x0=X0):
    """
    Expression analytique de la fonction d'onde (particule libre, ℏ=m=1).
    Formule exacte permettant de valider le code numérique.
    
    C'est la même formule que dans PaquetOndes.py (Compute_gaussian_wp),
    mais réécrite avec les paramètres de ce fichier et centrée sur x₀.
    """
    a2    = a**2
    terme = a2 + 2j * t                          # dénominateur complexe
    amp   = sqrt(a / (terme * numpy.sqrt(pi/2))) # amplitude normalisée
    
    # Phase et envelope (centrées sur x0 + v_g * t)
    v_g   = HBAR * k0 / MASS   # vitesse de groupe = k0 (avec ℏ=m=1)
    return amp * numpy.exp(-(x - x0 - v_g * t)**2 / (2 * terme)) * numpy.exp(1j * k0 * (x - x0))


def Comparer_numerique_analytique():
    """
    Section 3.2.5 : Confronte la simulation numérique avec la solution analytique.
    
    Si les deux courbes coïncident, notre algorithme est correct.
    C'est la VALIDATION de la méthode numérique.
    """
    print("\n=== Comparaison numérique vs analytique (particule libre V₀ = 0) ===")
    
    # Lance la simulation
    densites, temps, normes = Evoluer_schrodinger(V=None, save_every=400)
    
    # Instants à comparer (on prend quelques instants parmi ceux sauvegardés)
    indices_a_comparer = [0, len(temps)//3, 2*len(temps)//3, len(temps)-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    fig.suptitle("Validation : Numérique vs Analytique (V₀ = 0)", fontsize=13)
    
    for i, idx in enumerate(indices_a_comparer):
        t_val = temps[idx]
        
        # Solution analytique
        psi_ana  = GaussWP_analytique(x, t_val)
        dens_ana = numpy.abs(psi_ana)**2
        
        # Solution numérique (sauvegardée)
        dens_num = densites[idx]
        
        # Erreur L1 (intégrale de la valeur absolue de la différence)
        erreur = scipy.integrate.simpson(numpy.abs(dens_num - dens_ana), x=x)
        
        axes[i].plot(x, dens_ana, 'k-',  lw=2.0, label=f"Analytique")
        axes[i].plot(x, dens_num, 'r--', lw=1.5, label=f"Numérique")
        axes[i].set_title(f"t = {t_val:.2f}  |  erreur L1 = {erreur:.2e}", fontsize=10)
        axes[i].set_xlabel("x"); axes[i].set_ylabel(r"$|\psi|^2$")
        axes[i].legend(fontsize=8); axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(bottom=0)
        
        print(f"  t = {t_val:.2f} : erreur L1 = {erreur:.2e}, norme num = {normes[idx]:.5f}")
    
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Partie 3.1 : Algorithmes de dérivation ===")
    Test_derivee_premiere()
    Test_derivee_seconde()
    
    print("\n=== Partie 3.2 : Algorithme de Schrödinger (V₀ = 0) ===")
    Comparer_numerique_analytique()
