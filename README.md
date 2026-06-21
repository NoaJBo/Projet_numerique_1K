# Projet numérique 1K — Simulation 1D de l'équation de Schrödinger

Ce dépôt contient des scripts Python pour construire et simuler des paquets d'ondes 1D, étudier l'effet tunnel sur une barrière potentielle et valider la méthode numérique par rapport à des solutions analytiques.

Ce README fournit des instructions simples pour lancer le projet et décrit l'utilité de chaque fichier principal.

---

## Prérequis

- Python 3.8+ (recommandé 3.8–3.11)
- Modules Python : numpy, matplotlib, scipy

Exemple d'installation dans un environnement virtuel :

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows (PowerShell: .venv\Scripts\Activate.ps1)

pip install --upgrade pip
pip install numpy matplotlib scipy
```

Remarque : certains scripts configurent `matplotlib.use("Qt5Agg")` pour des affichages interactifs. Si tu es sur un serveur sans interface graphique, modifie ces fichiers ou exporte `MPLBACKEND=Agg` avant d'exécuter pour produire des fichiers images sans fenêtre.

---

## Structure et utilité des fichiers

Tous les scripts se trouvent dans le dossier `Projet/`.

- `Projet/algo_derivation1K.py`
  - Fonctions utilitaires pour les dérivées numériques (dérivée première et seconde).
  - Contient des tests/exemples qui comparent les dérivées numériques aux dérivées analytiques (f(x)=x^2).
  - Utilisé par les autres scripts pour calculer les opérateurs spatiaux (différences finies).

- `Projet/paquet_onde_gauss_1d1K.py`
  - Fournit la fonction analytique du paquet d'ondes gaussien (fonction `GaussWP(k0,a,x,t)`).
  - Gère t=0 et l'évolution analytique pour t≠0 (unités réduites hbar=m=1 par défaut).
  - Permet d'illustrer l'étalement et la densité analytique |ψ|².

- `Projet/onde_plane_1d1K.py`
  - Fonctions utilitaires pour ondes planes (`PlaneWave`) et fonctions de visualisation.
  - Scripts de démonstration pour afficher une onde plane et la superposition de trois ondes.

- `Projet/PaquetOndeGauss1d1K.py` et `Projet/paquet_onde_gauss_1d1K.py`
  - Attention : le dépôt contient plusieurs versions portant des noms proches. `paquet_onde_gauss_1d1K.py` est la version principale contenant la formule analytique générale.
  - `PaquetOndeGauss1d1K.py` peut contenir une implémentation simplifiée (t=0) et sert à génération/visualisation rapide.

- `Projet/simulation_schrodinger1K.py`
  - Composants pour construire un potentiel (barrière rectangulaire), générer le paquet d'ondes initial et faire la simulation avec stockage 2D (Re_2D, Im_2D).
  - Contient la boucle temporelle (schéma type semi-implicite / Verlet) et une animation basique via Matplotlib.
  - Produit la densité finale et peut calculer le coefficient de transmission.

- `Projet/validation_particule_libre1K.py`
  - Script de validation : compare la simulation numérique en cas de particule libre (V=0) avec la solution analytique du paquet gaussien fournie par `paquet_onde_gauss_1d1K.py`.
  - Trace et enregistre des figures de comparaison et affiche l'erreur maximale sur |ψ|^2.

- `Projet/etude_parametres1K.py`
  - Script d'étude paramétrique : fait varier la largeur de la barrière `a` et la hauteur/profondeur `V0` et calcule le coefficient de transmission T.
  - Utilise une version "silencieuse" (sans affichage) `simuler_silencieux` pour calculs batch et trace les résultats.

- `Projet/etude_temporelle1K.py`
  - Étude temporelle : stocke l'historique complet de la densité (matrice 2D) pour mesurer des temps de transit (entrée/sortie) à travers la barrière.
  - Produit des graphiques montrant l'influence de la largeur/hauteur de la barrière sur les temps de franchissement.

- `Projet/SchrodingerNumerique1K.py` et `Projet/SchrodingerBarriere1K.py`
  - Versions plus longues et commentées de l'implémentation numérique (peuvent contenir du code redondant). Utiles pour référence et lecture pédagogique.

- `Projet/validation_particule_libre1K.py`, `Projet/simulation_schrodinger1K.py` et les scripts d'étude produisent des fichiers image (.png) dans le répertoire courant.

---

## Exemples d'exécution (ordre recommandé)

1. Vérifier les dérivées numériques :

```bash
python Projet/algo_derivation1K.py
```

2. Visualiser le paquet gaussien analytique (optionnel) :

```bash
python Projet/paquet_onde_gauss_1d1K.py
```

3. Lancer la simulation avec animation (nécessite backend graphique) :

```bash
python Projet/simulation_schrodinger1K.py
```

4. Validation numérique vs analytique (particule libre) :

```bash
python Projet/validation_particule_libre1K.py
```

5. Études paramétriques (sans affichage interactif) :

```bash
python Projet/etude_parametres1K.py
python Projet/etude_temporelle1K.py
```

Astuce : si tu exécutes sur un serveur sans affichage, définis le backend Matplotlib à `Agg` avant ou modifie les fichiers qui appellent `matplotlib.use("Qt5Agg")`. Exemple :

```bash
export MPLBACKEND=Agg
python Projet/etude_parametres1K.py
```

---

## Paramètres importants

- Unités : la plupart des scripts utilisent les unités réduites ℏ = m = 1 (indiqué dans les fichiers concernés). Vérifie les entêtes si tu modifies hbar/m.
- Grilles et pas : NX, NT, DX, DT sont définis localement dans les scripts; attention à la condition de stabilité numérique (s = Δt/(2Δx^2) pour certains schémas) — vérifie les messages imprimés au lancement.

---

## Conseils & améliorations possibles

- Pour automatiser les tests et tracer sans interface, utilise `MPLBACKEND=Agg` ou modifie `matplotlib.use(...)` vers `'Agg'` dans les scripts non interactifs.
- Si tu veux réutiliser le code comme module, extrais les fonctions (ex. `Compute_initial_wavepacket`, `Evoluer_schrodinger`) et importe‑les depuis un script d'usage.

---

## Prochaines améliorations possibles

- Ouvrir une Pull Request pour revue des modifications récentes.
- Ajouter un script `run_all.sh` pour exécuter la suite de validation et sauvegarder tous les graphiques.
- Ajouter tests unitaires simples (par ex. vérification de la conservation de la norme sur une courte simulation).

---

Si tu veux que j'ajoute ce README dans le dépôt, confirme et je le committerai sur la branche principale.
