####------SchrodingerBarriere-----####
# Partie 4 du projet : Effet tunnel et temps de traversée
# Résolution numérique de Schrödinger avec barrière rectangulaire
# Unités naturelles : ℏ = m = 1
##

import numpy
from numpy import pi, exp, sqrt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate

# ════════════════════════════════════════════════════════════════════════════════
#  CONSTANTES ET PARAMÈTRES
# ════════════════════════════════════════════════════════════════════════════════

HBAR = 1.0
MASS = 1.0

# Paramètres du paquet d'ondes gaussien initial
K0    = 2.0     # nombre d'onde central → impulsion moyenne p₀ = ℏk₀ = k₀
SIGMA = 0.3     # largeur en k (= 1/a dans la notation de l'énoncé)
A_WP  = 1.0 / SIGMA   # largeur réelle du paquet (paramètre a)
X0    = -20.0   # position initiale du centre du paquet

# Énergie cinétique moyenne : E₀ = ℏ²k₀²/(2m) = k₀²/2 (avec ℏ=m=1)
E0 = HBAR**2 * K0**2 / (2 * MASS)

# Paramètres de la barrière (valeurs par défaut, modifiables)
X_BARRIERE = 0.0    # bord gauche de la barrière
A_BARRIERE = 3.0    # largeur de la barrière (variable dans l'étude)
V0         = 3.0    # hauteur de la barrière (en régime tunnel : V0 > E0)

print(f"Énergie cinétique moyenne : E₀ = k₀²/2 = {E0:.2f}")
print(f"Hauteur de barrière : V₀ = {V0:.2f}")
if V0 > E0:
    print(f"→ Régime TUNNEL (E₀ < V₀) : l'effet tunnel est actif !")
else:
    print(f"→ Régime CLASSIQUE (E₀ > V₀) : passage classique attendu")

# Grille spatiale
X_MIN, X_MAX = -40.0, 60.0
N_X          = 1500
DX           = (X_MAX - X_MIN) / (N_X - 1)
x            = numpy.linspace(X_MIN, X_MAX, N_X)

# Grille temporelle
T_MAX  = 50.0
DT     = 0.001
N_T    = int(T_MAX / DT)

# Coefficient de stabilité (doit être < 0.25)
S = HBAR * DT / (2.0 * MASS * DX**2)
print(f"\nStabilité : s = {S:.5f}  {'✓' if S < 0.25 else '✗ ATTENTION'}")


# ════════════════════════════════════════════════════════════════════════════════
#  FONCTIONS DE BASE
# ════════════════════════════════════════════════════════════════════════════════

def Build_barrier(x, x_bar=X_BARRIERE, a_bar=A_BARRIERE, V0=V0):
    """
    Construit le potentiel rectangulaire V(x).
    
    V(x) = V0  si  x_bar ≤ x ≤ x_bar + a_bar
    V(x) = 0   sinon
    
    C'est la modélisation d'un obstacle pour la particule.
    La barrière a une hauteur V0 et une largeur a_bar.
    """
    V = numpy.zeros_like(x)
    # numpy booléen indexing : sélectionne les points à l'intérieur de la barrière
    masque = (x >= x_bar) & (x <= x_bar + a_bar)
    V[masque] = V0
    return V


def Build_initial_wavepacket(x, k0=K0, sigma=SIGMA, x0=X0):
    """
    Paquet d'ondes gaussien initial normalisé.
    
    Ψ(x, 0) = (σ/√π)^(1/2) × exp(-σ²(x-x₀)²/2) × exp(ik₀(x-x₀))
    
    La normalisation assure ∫|Ψ|²dx = 1.
    """
    facteur_norm = (sigma / numpy.sqrt(pi))**0.5
    gaussienne   = numpy.exp(-sigma**2 * (x - x0)**2 / 2.0)
    porteuse     = numpy.exp(1j * k0 * (x - x0))
    return facteur_norm * gaussienne * porteuse


def Check_normalization(psi_or_density, x):
    """Retourne ∫|Ψ|²dx ou ∫ρdx si on passe la densité directement."""
    if numpy.iscomplexobj(psi_or_density):
        densite = numpy.abs(psi_or_density)**2
    else:
        densite = psi_or_density
    return scipy.integrate.simpson(densite, x=x)


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 4.1a — Algorithme de résolution de Schrödinger avec barrière
# ════════════════════════════════════════════════════════════════════════════════

def Evoluer_avec_barriere(V, save_every=500):
    """
    Fait évoluer la fonction d'onde en présence de la barrière de potentiel.
    
    Même algorithme que dans SchrodingerNumerique.py (méthode R/I semi-implicite),
    mais avec le potentiel V(x) ≠ 0 dans la région de la barrière.
    
    Paramètres:
      V          : tableau du potentiel, shape (N_X,)
      save_every : fréquence de sauvegarde (en pas de temps)
    
    Retourne:
      densites      : |Ψ(x,tⱼ)|² pour chaque instant sauvegardé
      temps         : instants sauvegardés
      normes        : norme ∫|Ψ|²dx à chaque instant
      R_finaux      : partie réelle à t_max
      I_finaux      : partie imaginaire à t_max
      positions_max : position du maximum de la densité à chaque instant
    """
    psi0 = Build_initial_wavepacket(x)
    R = numpy.real(psi0).copy()
    I = numpy.imag(psi0).copy()
    
    densites      = []
    temps         = []
    normes        = []
    positions_max = []
    
    # Sauvegarde initiale
    densite0 = R**2 + I**2
    densites.append(densite0.copy())
    temps.append(0.0)
    normes.append(scipy.integrate.simpson(densite0, x=x))
    positions_max.append(x[numpy.argmax(densite0)])
    
    for j in range(N_T):
        # Mise à jour de I (partie imaginaire) avec R actuel
        I[1:-1] = (
            I[1:-1]
            - S * (R[2:] + R[:-2] - 2.0 * R[1:-1])
            + DT / HBAR * V[1:-1] * R[1:-1]
        )
        
        # Mise à jour de R (partie réelle) avec le nouveau I
        R[1:-1] = (
            R[1:-1]
            + S * (I[2:] + I[:-2] - 2.0 * I[1:-1])
            - DT / HBAR * V[1:-1] * I[1:-1]
        )
        
        # Conditions aux bords absorbantes
        R[0]  = R[-1]  = 0.0
        I[0]  = I[-1]  = 0.0
        
        if j % save_every == 0:
            densite = R**2 + I**2
            densites.append(densite.copy())
            temps.append((j + 1) * DT)
            normes.append(scipy.integrate.simpson(densite, x=x))
            positions_max.append(x[numpy.argmax(densite)])
    
    return (numpy.array(densites),
            numpy.array(temps),
            numpy.array(normes),
            R.copy(), I.copy(),
            numpy.array(positions_max))


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 4.1b — Mesure de τ₀,num (particule libre, V₀ = 0)
# ════════════════════════════════════════════════════════════════════════════════

def Mesurer_tau0_numerique(a=A_BARRIERE):
    """
    Mesure τ₀,num = temps mis par la particule LIBRE pour parcourir la distance a.
    
    Méthode :
      On suit la position du maximum de la densité |Ψ|² au cours du temps.
      τ₀,num = temps pour que le maximum passe de X0 à X0+a.
    
    Comparaison avec la théorie :
      τ₀,th = a / v_g = a × m / (ℏ × k₀) = a/k₀  (avec ℏ=m=1)
    
    Paramètre:
      a : distance à parcourir (= largeur de la barrière pour comparaison)
    """
    print(f"\n=== Mesure de τ₀,num (particule libre, a = {a}) ===")
    
    # Potentiel nul (particule libre)
    V_libre = numpy.zeros(N_X)
    
    _, temps, _, _, _, positions_max = Evoluer_avec_barriere(V_libre, save_every=100)
    
    # Position initiale du maximum = X0
    # On cherche quand le max atteint X0 + a
    x_depart = X0
    x_arrivee = X0 + a
    
    # Trouver l'instant où la position dépasse x_arrivee
    tau0_num = None
    for i, pos in enumerate(positions_max):
        if pos >= x_arrivee:
            # Interpolation linéaire entre les deux instants encadrants
            if i > 0:
                t1, t2 = temps[i-1], temps[i]
                p1, p2 = positions_max[i-1], positions_max[i]
                tau0_num = t1 + (x_arrivee - p1) / (p2 - p1) * (t2 - t1)
            else:
                tau0_num = temps[i]
            break
    
    # Valeur théorique : τ₀,th = a / v_g = a / (ℏk₀/m) = a/k₀
    v_g = HBAR * K0 / MASS   # vitesse de groupe
    tau0_th = a / v_g
    
    print(f"  Vitesse de groupe : v_g = ℏk₀/m = {v_g:.3f}")
    print(f"  τ₀,th (analytique) = a/v_g = {tau0_th:.4f}")
    print(f"  τ₀,num (simulation) = {tau0_num:.4f}" if tau0_num else "  τ₀,num : non atteint (augmenter T_MAX)")
    if tau0_num:
        err_rel = abs(tau0_num - tau0_th) / tau0_th * 100
        print(f"  Écart relatif : {err_rel:.2f}%")
    
    return tau0_num, tau0_th, temps, positions_max


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 4.1c — Mesure de τt,num (avec barrière, effet tunnel)
# ════════════════════════════════════════════════════════════════════════════════

def Mesurer_taut_numerique(a=A_BARRIERE, v0=V0):
    """
    Mesure τt,num = temps de traversée de la barrière par effet tunnel.
    
    MÉTHODE : on suit le maximum de la densité TRANSMISE (à droite de la barrière).
    
    Le paquet se divise en deux à la barrière :
      - Un paquet réfléchi (va vers la gauche)
      - Un paquet transmis (émerge à droite)
    
    On mesure le temps entre :
      t₁ = instant où le centre du paquet incident arrive au bord gauche de la barrière
      t₂ = instant où le maximum du paquet transmis atteint le bord droit
    
    τt,num = t₂ - t₁
    
    Paramètres:
      a  : largeur de la barrière
      v0 : hauteur de la barrière
    """
    print(f"\n=== Mesure de τt,num (avec barrière, a = {a}, V₀ = {v0}) ===")
    print(f"  E₀ = k₀²/2 = {E0:.2f}")
    if v0 > E0:
        kappa = numpy.sqrt(2 * MASS * (v0 - E0)) / HBAR
        print(f"  Régime tunnel : κ = √(2m(V₀-E₀))/ℏ = {kappa:.3f}")
        print(f"  Facteur tunnel : exp(-2κa) = {numpy.exp(-2*kappa*a):.4e}")
    else:
        print(f"  Régime classique (E₀ > V₀)")
    
    # Construction de la barrière
    V = Build_barrier(x, x_bar=X_BARRIERE, a_bar=a, V0=v0)
    
    # Simulation
    densites, temps, normes, R_fin, I_fin, positions_max = Evoluer_avec_barriere(
        V, save_every=200
    )
    
    # ── Point de référence DROIT : juste après la barrière ──
    # On y mesure la densité pour détecter le paquet transmis
    x_ref_gauche = X_BARRIERE              # bord gauche de la barrière
    x_ref_droite = X_BARRIERE + a          # bord droit de la barrière
    x_ref_mesure = X_BARRIERE + a + 0.5   # point juste après la barrière
    
    # Indice du point de mesure dans le tableau x
    idx_mesure = numpy.argmin(numpy.abs(x - x_ref_mesure))
    
    # ── Détection du temps d'arrivée du paquet incident à x_gauche ──
    # t₁ = quand le max de la densité atteint le bord gauche de la barrière
    t1 = None
    for i, pos in enumerate(positions_max):
        if pos >= x_ref_gauche:
            if i > 0:
                t2_arr, t1_arr = temps[i], temps[i-1]
                p2_arr, p1_arr = positions_max[i], positions_max[i-1]
                t1 = t1_arr + (x_ref_gauche - p1_arr) / (p2_arr - p1_arr) * (t2_arr - t1_arr)
            else:
                t1 = temps[i]
            break
    
    # ── Détection du temps d'arrivée du paquet transmis à x_droite ──
    # On cherche quand la densité au point de mesure atteint son maximum
    # C'est le moment où le centre du paquet transmis passe par ce point
    densite_au_point = densites[:, idx_mesure]
    
    # On cherche le maximum de la densité transmise APRÈS que le paquet soit arrivé
    # (pour ne pas confondre avec la densité initiale = 0)
    seuil = 0.1 * numpy.max(densite_au_point)   # seuil = 10% du max
    
    t2 = None
    # On cherche le maximum global de la densité transmise
    if numpy.max(densite_au_point) > 1e-10:  # si transmission détectable
        idx_max_transmis = numpy.argmax(densite_au_point)
        t2 = temps[idx_max_transmis]
    
    # ── Résultats ──
    print(f"\n  t₁ (arrivée au bord gauche) = {t1:.4f}" if t1 else "  t₁ : non détecté")
    print(f"  t₂ (paquet transmis au bord droit) = {t2:.4f}" if t2 else "  t₂ : non détecté (T trop petit ou transmission nulle)")
    
    tau_t_num = None
    if t1 and t2:
        tau_t_num = t2 - t1
        tau0_th   = a / (HBAR * K0 / MASS)   # temps classique de référence
        print(f"\n  τt,num = t₂ - t₁ = {tau_t_num:.4f}")
        print(f"  τ₀,th  = a/v_g   = {tau0_th:.4f}  (temps classique, V₀=0)")
        if tau_t_num < tau0_th:
            print(f"  → Le paquet tunnel est PLUS RAPIDE que le classique ! (effet Hartman)")
        else:
            print(f"  → Le paquet tunnel est plus lent que le classique.")
    
    return tau_t_num, densites, temps, normes, V


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 4.1d — Influence de la largeur a sur τ₀,num et τt,num
# ════════════════════════════════════════════════════════════════════════════════

def Etude_influence_a(liste_a=None, v0=V0):
    """
    Étudie comment τ₀,num et τt,num varient avec la largeur a de la barrière.
    
    Attendu théoriquement :
      τ₀,th = a/v_g  → croît LINÉAIREMENT avec a (la particule libre parcourt a)
      τt,th          → sature avec a (effet Hartman : temps tunnel indépendant de a grands)
    """
    if liste_a is None:
        liste_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    print(f"\n=== Étude influence de a (V₀ = {v0}, E₀ = {E0:.2f}) ===")
    
    tau0_th_liste  = []
    taut_num_liste = []
    v_g = HBAR * K0 / MASS
    
    for a in liste_a:
        print(f"\n--- a = {a} ---")
        tau0_th = a / v_g
        tau0_th_liste.append(tau0_th)
        
        tau_t, _, _, _, _ = Mesurer_taut_numerique(a=a, v0=v0)
        taut_num_liste.append(tau_t if tau_t else float('nan'))
    
    # ── Graphique ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(liste_a, tau0_th_liste, 'k-o', lw=2, ms=7, label=r"$\tau_{0,th} = a/v_g$ (particule libre)")
    
    valides = [(a, tau) for a, tau in zip(liste_a, taut_num_liste) if not numpy.isnan(tau)]
    if valides:
        a_vals, tau_vals = zip(*valides)
        ax.plot(a_vals, tau_vals, 'rs--', lw=1.8, ms=8, label=r"$\tau_{t,num}$ (tunnel numérique)")
    
    ax.set_xlabel("Largeur de la barrière a", fontsize=12)
    ax.set_ylabel("Temps de traversée τ", fontsize=12)
    ax.set_title(f"Influence de a sur τ  |  V₀ = {v0}, k₀ = {K0}", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
    
    # Annotation sur l'effet Hartman
    ax.annotate(
        "Saturation attendue\n(effet Hartman)",
        xy=(liste_a[-1], taut_num_liste[-1] if not numpy.isnan(taut_num_liste[-1]) else 1),
        xytext=(liste_a[-1] * 0.5, max(tau0_th_liste) * 0.3),
        arrowprops=dict(arrowstyle='->', color='red'),
        fontsize=9, color='red'
    )
    
    plt.tight_layout()
    plt.show()
    
    return liste_a, tau0_th_liste, taut_num_liste


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 4.1e — Influence de V₀ sur τt,num
# ════════════════════════════════════════════════════════════════════════════════

def Etude_influence_V0(liste_V0=None, a=A_BARRIERE):
    """
    Étudie comment τt,num varie avec la hauteur V₀ de la barrière.
    
    Attendu théoriquement :
      Plus V₀ augmente → κ = √(2m(V₀-E₀))/ℏ augmente → transmission T ↓
      Le paquet transmis est de plus en plus difficile à localiser.
      Le temps de traversée varie selon le modèle de temps de phase.
    """
    if liste_V0 is None:
        # On prend des valeurs autour de E₀ (en dessous, autour, et au-dessus)
        liste_V0 = [E0 * 0.8, E0, E0 * 1.5, E0 * 2.0, E0 * 3.0]
    
    print(f"\n=== Étude influence de V₀ (a = {a}, E₀ = {E0:.2f}) ===")
    
    taut_num_liste = []
    T_coeff_liste  = []   # coefficient de transmission
    
    for v0 in liste_V0:
        print(f"\n--- V₀ = {v0:.2f} ---")
        
        # Coefficient de transmission analytique (formule des états stationnaires)
        T_analytic = Compute_transmission_coeff(K0, v0, a)
        T_coeff_liste.append(T_analytic)
        print(f"  T(k₀) = {T_analytic:.4e}")
        
        tau_t, _, _, _, _ = Mesurer_taut_numerique(a=a, v0=v0)
        taut_num_liste.append(tau_t if tau_t else float('nan'))
    
    # Graphique
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Influence de V₀  |  a = {a}, k₀ = {K0}", fontsize=12)
    
    valides = [(v, tau) for v, tau in zip(liste_V0, taut_num_liste) if not numpy.isnan(tau)]
    if valides:
        v_vals, tau_vals = zip(*valides)
        ax1.plot(v_vals, tau_vals, 'bs-', lw=1.8, ms=8, label=r"$\tau_{t,num}$")
    ax1.axvline(E0, color='gray', ls='--', lw=1.5, label=f"$E_0 = {E0:.2f}$")
    ax1.set_xlabel("Hauteur de la barrière V₀", fontsize=11)
    ax1.set_ylabel(r"Temps de traversée $\tau_{t,num}$", fontsize=11)
    ax1.set_title("Temps de traversée vs V₀")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.4)
    
    ax2.semilogy(liste_V0, T_coeff_liste, 'ro-', lw=1.8, ms=7, label="T (analytique)")
    ax2.axvline(E0, color='gray', ls='--', lw=1.5, label=f"$E_0 = {E0:.2f}$")
    ax2.set_xlabel("Hauteur de la barrière V₀", fontsize=11)
    ax2.set_ylabel("Coefficient de transmission T (log)", fontsize=11)
    ax2.set_title("Transmission vs V₀ (échelle log)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.show()
    
    return liste_V0, taut_num_liste, T_coeff_liste


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 4.3a — Calcul analytique du coefficient de transmission
# ════════════════════════════════════════════════════════════════════════════════

def Compute_transmission_coeff(k0=K0, v0=V0, a=A_BARRIERE):
    """
    Calcule le coefficient de transmission T à partir des états stationnaires.
    
    États stationnaires pour la barrière rectangulaire (E = ℏ²k²/2m) :
    
    Zone I  (x < 0)       : ψ = A·exp(ikx) + B·exp(-ikx)       onde incidente + réfléchie
    Zone II (0 < x < a)   : ψ = C·exp(κx) + D·exp(-κx)         onde évanescente (E < V₀)
    Zone III (x > a)      : ψ = F·exp(ikx)                      onde transmise
    
    Avec :
      k = √(2mE)/ℏ = k₀  (avec ℏ=m=1)
      κ = √(2m(V₀-E))/ℏ = √(V₀ - k₀²/2) × √2  (si E < V₀, régime tunnel)
      (si E > V₀ : κ = ik' avec k' = √(2m(E-V₀))/ℏ, formule différente)
    
    Les conditions de continuité en x=0 et x=a donnent T = |F/A|² :
    
    En régime TUNNEL (E < V₀) :
      T = 1 / (1 + (k² + κ²)²·sinh²(κa) / (4k²κ²))
    
    En régime CLASSIQUE (E > V₀) :
      T = 1 / (1 + (k² - k'²)²·sin²(k'a) / (4k²k'²))
    """
    k = k0   # nombre d'onde (avec ℏ=m=1, E = k²/2, donc k = √(2E) = k₀)
    E = HBAR**2 * k**2 / (2 * MASS)   # = k₀²/2
    
    if v0 > E:
        # ── Régime tunnel (E < V₀) ──
        # κ est réel → exponentielle décroissante dans la barrière
        kappa = numpy.sqrt(2 * MASS * (v0 - E)) / HBAR   # = √(2(V₀ - k₀²/2))
        
        # sinh(κa) peut devenir très grand → on utilise des fonctions numpy stables
        sh = numpy.sinh(kappa * a)
        
        # Formule des états stationnaires
        T = 1.0 / (1.0 + (k**2 + kappa**2)**2 * sh**2 / (4.0 * k**2 * kappa**2))
    else:
        # ── Régime classique (E > V₀) ──
        # k' est réel → oscillations dans la barrière
        k_prime = numpy.sqrt(2 * MASS * (E - v0)) / HBAR
        
        si = numpy.sin(k_prime * a)
        
        # Formule analogue avec k' et sin
        if k_prime > 0:
            T = 1.0 / (1.0 + (k**2 - k_prime**2)**2 * si**2 / (4.0 * k**2 * k_prime**2))
        else:
            T = 1.0   # k' = 0 → barrière transparente exactement à E = V₀
    
    return T


def Compute_reflection_coeff(k0=K0, v0=V0, a=A_BARRIERE):
    """
    Coefficient de réflexion R = 1 - T.
    Vérifie la conservation de la probabilité : R + T = 1.
    """
    T = Compute_transmission_coeff(k0, v0, a)
    R = 1.0 - T
    return R


# ════════════════════════════════════════════════════════════════════════════════
#  VISUALISATION — Animation et graphiques
# ════════════════════════════════════════════════════════════════════════════════

def Visualiser_effet_tunnel(densites, temps, normes, V, titre="Effet tunnel"):
    """
    Affiche plusieurs snapshots de l'évolution du paquet d'ondes
    face à la barrière. Montre le paquet se séparer en deux.
    """
    n_snapshots = min(6, len(densites))
    # On choisit les instants à afficher uniformément
    indices = numpy.linspace(0, len(densites)-1, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    fig.suptitle(titre, fontsize=13)
    
    y_max = numpy.max(densites) * 1.2
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.plot(x, densites[idx], color='steelblue', lw=1.8, label=r"$|\psi|^2$")
        
        # Afficher la barrière (en transparence)
        ax.fill_between(x, 0, V / numpy.max(V) * y_max * 0.3,
                        where=(V > 0), alpha=0.3, color='orange', label="Barrière")
        
        ax.set_title(f"t = {temps[idx]:.2f}  |  norme = {normes[idx]:.4f}", fontsize=9)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel(r"$|\psi|^2$", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()


def Animer_effet_tunnel(densites, temps, normes, V):
    """
    Animation de l'évolution du paquet d'ondes.
    Montre en temps réel le paquet arriver sur la barrière, se diviser
    en paquet réfléchi (allant à gauche) et paquet transmis (allant à droite).
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Animation : effet tunnel quantique", fontsize=12)
    
    y_max = numpy.max(densites) * 1.2
    
    # Représenter la barrière (normalisée pour l'affichage)
    ax.fill_between(x, 0, V / numpy.max(V) * y_max * 0.4 if numpy.max(V) > 0 else V,
                    where=(V > 0), alpha=0.25, color='orange', label="Barrière V(x)")
    
    (ligne,) = ax.plot(x, densites[0], color='steelblue', lw=2.0, label=r"$|\psi(x,t)|^2$")
    temps_texte = ax.text(0.02, 0.93, "", transform=ax.transAxes, fontsize=10)
    norme_texte = ax.text(0.02, 0.85, "", transform=ax.transAxes, fontsize=9, color='gray')
    
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("x"); ax.set_ylabel(r"$|\psi|^2$")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    
    def update(frame):
        ligne.set_ydata(densites[frame])
        temps_texte.set_text(f"t = {temps[frame]:.2f}")
        norme_texte.set_text(f"norme = {normes[frame]:.4f}")
        return ligne, temps_texte, norme_texte
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(densites),
        interval=40, blit=True, repeat=True
    )
    plt.show()
    return ani


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN — Orchestration de toutes les études
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("  PROJET : EFFET TUNNEL — RÉSULTATS")
    print("="*60)
    
    # ── Coefficient de transmission analytique ─────────────────────────────
    T = Compute_transmission_coeff(K0, V0, A_BARRIERE)
    R_coeff = Compute_reflection_coeff(K0, V0, A_BARRIERE)
    print(f"\n[Analytique] T = {T:.4e}, R = {R_coeff:.4f}, T+R = {T+R_coeff:.6f}")
    
    # ── 4.1a : Simulation et visualisation de l'effet tunnel ──────────────
    print("\n[4.1a] Simulation de l'effet tunnel...")
    V = Build_barrier(x, X_BARRIERE, A_BARRIERE, V0)
    densites, temps, normes, R_fin, I_fin, positions_max = Evoluer_avec_barriere(
        V, save_every=300
    )
    
    # Snapshots de l'évolution
    Visualiser_effet_tunnel(densites, temps, normes, V,
                            titre=f"Effet tunnel | V₀={V0}, a={A_BARRIERE}, k₀={K0}")
    
    # Animation (commenter si trop long)
    Animer_effet_tunnel(densites, temps, normes, V)
    
    # ── 4.1b : τ₀,num (particule libre) ───────────────────────────────────
    print("\n[4.1b] Mesure de τ₀,num...")
    tau0_num, tau0_th, _, _ = Mesurer_tau0_numerique(a=A_BARRIERE)
    
    # ── 4.1c : τt,num (avec barrière) ─────────────────────────────────────
    print("\n[4.1c] Mesure de τt,num...")
    tau_t, _, _, _, _ = Mesurer_taut_numerique(a=A_BARRIERE, v0=V0)
    
    # ── 4.1d : Influence de a ──────────────────────────────────────────────
    print("\n[4.1d] Étude influence de la largeur a...")
    Etude_influence_a(liste_a=[1.0, 2.0, 3.0, 4.0, 5.0], v0=V0)
    
    # ── 4.1e : Influence de V₀ ─────────────────────────────────────────────
    print("\n[4.1e] Étude influence de la hauteur V₀...")
    Etude_influence_V0(liste_V0=[E0*0.5, E0*1.0, E0*1.5, E0*2.0, E0*3.0],
                       a=A_BARRIERE)
    
    print("\n" + "="*60)
    print("  FIN DE LA SIMULATION")
    print("="*60)
