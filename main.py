"""
Résolution du problème de transport - Méthode des potentiels (MODI)
Usage : python main.py [fichier.txt]
"""

import sys
import time
import random
from collections import deque

# ─────────────────────────────────────────────
# 1. LECTURE / GÉNÉRATION
# ─────────────────────────────────────────────

def lire_fichier(chemin):
    """Lit un fichier .txt au format :
        n m
        c[0][0] ... c[0][m-1]  P[0]
        ...
        c[n-1][0] ... c[n-1][m-1]  P[n-1]
        D[0] ... D[m-1]
    Retourne (n, m, couts, provisions, demandes).
    """
    with open(chemin) as f:
        lignes = [l.strip() for l in f if l.strip()]
    n, m = map(int, lignes[0].split())
    couts = []
    provisions = []
    for i in range(1, n + 1):
        vals = list(map(int, lignes[i].split()))
        couts.append(vals[:m])
        provisions.append(vals[m])
    demandes = list(map(int, lignes[n + 1].split()))
    return n, m, couts, provisions, demandes


def generer_aleatoire(n, m):
    """Génère un problème équilibré selon la méthode de la matrice temp (consignes §3.3.1) :
    temp[i][j] ∈ [1,100]  →  P_i = somme ligne i,  C_j = somme colonne j.
    Garantit ΣP_i = ΣC_j sans ajustement.
    """
    couts = [[random.randint(1, 100) for _ in range(m)] for _ in range(n)]
    temp  = [[random.randint(1, 100) for _ in range(m)] for _ in range(n)]
    provisions = [sum(temp[i])    for i in range(n)]
    demandes   = [sum(temp[i][j] for i in range(n)) for j in range(m)]
    return n, m, couts, provisions, demandes


# ─────────────────────────────────────────────
# 2. AFFICHAGE
# ─────────────────────────────────────────────

def _col_width(n, m, couts, alloc=None):
    """Calcule la largeur de colonne pour un bel alignement."""
    w = 4
    for i in range(n):
        for j in range(m):
            w = max(w, len(str(couts[i][j])))
            if alloc:
                w = max(w, len(str(alloc[i][j])))
    return w + 2


def afficher_matrice_couts(n, m, couts, provisions, demandes):
    w = _col_width(n, m, couts)
    print("\n=== Matrice des coûts ===")
    header = "     " + "".join(f"D{j+1}".center(w) for j in range(m)) + "  Pi"
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"O{i+1} |" + "".join(str(couts[i][j]).center(w) for j in range(m))
        row += f"  {provisions[i]}"
        print(row)
    print("-" * len(header))
    print("Cj  " + "".join(str(demandes[j]).center(w) for j in range(m)))


def afficher_matrice_transport(n, m, couts, alloc, provisions, demandes):
    w = _col_width(n, m, couts, alloc)
    print("\n=== Matrice de transport ===")
    header = "     " + "".join(f"D{j+1}".center(w) for j in range(m)) + "  Pi"
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"O{i+1} |"
        for j in range(m):
            if alloc[i][j] is not None:
                row += str(alloc[i][j]).center(w)
            else:
                row += "-".center(w)
        row += f"  {provisions[i]}"
        print(row)
    print("-" * len(header))
    print("Cj  " + "".join(str(demandes[j]).center(w) for j in range(m)))


def afficher_potentiels_et_marginaux(n, m, couts, alloc, potS, potT):
    """Affiche la table des coûts potentiels et des coûts marginaux."""
    w = 7
    print("\n=== Coûts potentiels  c̃(i,j) = E(i) + E(j) ===")
    print("     " + "".join(f"D{j+1}({potT[j]:+d})".center(w) for j in range(m)))
    for i in range(n):
        row = f"O{i+1}({potS[i]:+d})|"
        for j in range(m):
            pot = potS[i] + potT[j]
            row += str(pot).center(w)
        print(row)

    print("\n=== Coûts marginaux  d(i,j) = c(i,j) - c̃(i,j) ===")
    print("     " + "".join(f"D{j+1}".center(w) for j in range(m)))
    for i in range(n):
        row = f"O{i+1} |"
        for j in range(m):
            if alloc[i][j] is not None:
                row += " BASE ".center(w)
            else:
                d = couts[i][j] - (potS[i] + potT[j])
                row += str(d).center(w)
        print(row)


# ─────────────────────────────────────────────
# 3. SOLUTION INITIALE
# ─────────────────────────────────────────────

def _init_alloc(n, m):
    return [[None] * m for _ in range(n)]


def nord_ouest(n, m, provisions, demandes):
    """Méthode du coin Nord-Ouest."""
    alloc = _init_alloc(n, m)
    p = provisions[:]
    d = demandes[:]
    i = j = 0
    while i < n and j < m:
        q = min(p[i], d[j])
        alloc[i][j] = q
        p[i] -= q
        d[j] -= q
        if p[i] == 0 and d[j] == 0:
            # dégénérescence : on ajoute un 0 sur l'arête suivante
            if i + 1 < n and j + 1 < m:
                alloc[i + 1][j] = 0   # marqueur, sera corrigé si besoin
            i += 1
            j += 1
        elif p[i] == 0:
            i += 1
        else:
            j += 1
    return alloc


def balas_hammer(n, m, couts, provisions, demandes):
    """Méthode de Vogel (Balas-Hammer) avec affichage des pénalités."""
    alloc = _init_alloc(n, m)
    p = provisions[:]
    d = demandes[:]
    active_rows = list(range(n))
    active_cols = list(range(m))

    iteration = 1
    while active_rows and active_cols:
        print(f"\n  [BH itération {iteration}]")
        pen_rows = {}
        pen_cols = {}

        for i in active_rows:
            vals = sorted(couts[i][j] for j in active_cols)
            pen_rows[i] = vals[1] - vals[0] if len(vals) >= 2 else vals[0]

        for j in active_cols:
            vals = sorted(couts[i][j] for i in active_rows)
            pen_cols[j] = vals[1] - vals[0] if len(vals) >= 2 else vals[0]

        # Affichage pénalités
        print("    Pénalités lignes : " + ", ".join(f"L{i+1}={pen_rows[i]}" for i in active_rows))
        print("    Pénalités colonnes: " + ", ".join(f"C{j+1}={pen_cols[j]}" for j in active_cols))

        # Choix : max pénalité
        best_val = -1
        choice = ('row', 0)
        for i in active_rows:
            if pen_rows[i] > best_val:
                best_val, choice = pen_rows[i], ('row', i)
        for j in active_cols:
            if pen_cols[j] > best_val:
                best_val, choice = pen_cols[j], ('col', j)

        if choice[0] == 'row':
            i = choice[1]
            j = min(active_cols, key=lambda jj: couts[i][jj])
        else:
            j = choice[1]
            i = min(active_rows, key=lambda ii: couts[ii][j])

        q = min(p[i], d[j])
        alloc[i][j] = q
        p[i] -= q
        d[j] -= q
        print(f"    → Affectation alloc[{i+1}][{j+1}] = {q}")

        if p[i] == 0:
            active_rows.remove(i)
        if d[j] == 0:
            active_cols.remove(j)
        iteration += 1

    return alloc


# ─────────────────────────────────────────────
# 4. COÛT TOTAL
# ─────────────────────────────────────────────

def cout_total(n, m, couts, alloc):
    return sum(couts[i][j] * alloc[i][j]
               for i in range(n) for j in range(m)
               if alloc[i][j] is not None and alloc[i][j] > 0)


# ─────────────────────────────────────────────
# 5. GRAPHE BIPARTI (BFS)
# ─────────────────────────────────────────────
# Nœuds : sources 0..n-1, destinations n..n+m-1

def _aretes(n, m, alloc):
    """Retourne la liste des arêtes de base (i, j) avec alloc[i][j] non None."""
    return [(i, j) for i in range(n) for j in range(m) if alloc[i][j] is not None]


def est_connexe(n, m, alloc):
    """BFS sur le graphe biparti – retourne True si connexe."""
    aretes = _aretes(n, m, alloc)
    if not aretes:
        return False
    adj = {k: [] for k in range(n + m)}
    for (i, j) in aretes:
        adj[i].append(n + j)
        adj[n + j].append(i)

    visited = set()
    q = deque([0])
    visited.add(0)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                q.append(v)

    noeuds_utiles = set()
    for (i, j) in aretes:
        noeuds_utiles.add(i)
        noeuds_utiles.add(n + j)
    return noeuds_utiles <= visited


def detecter_cycle(n, m, alloc):
    """DFS – retourne le cycle sous forme [(i,j),...] ou None.
    Un arbre couvrant d'un graphe biparti à k arêtes doit avoir k = n+m-1 arêtes.
    Si k > n+m-1 il y a forcément un cycle.
    On cherche le cycle en DFS.
    """
    aretes = set(_aretes(n, m, alloc))
    adj = {k: [] for k in range(n + m)}
    for (i, j) in aretes:
        adj[i].append((n + j, (i, j)))
        adj[n + j].append((i, (i, j)))

    # Approche robuste : compter les arêtes vs n+m-1
    nb_aretes = len(aretes)
    nb_noeuds = len({i for i, _ in aretes} | {n + j for _, j in aretes})
    if nb_aretes < nb_noeuds - 1:
        return None   # pas encore d'info sur cycle (non connexe géré ailleurs)
    if nb_aretes == nb_noeuds - 1:
        return None   # arbre = pas de cycle

    # nb_aretes > nb_noeuds - 1 → cycle
    # On cherche l'arête redondante par DFS
    visited2 = set()
    parent2  = {}   # nœud -> nœud parent dans l'arbre DFS
    edge_par = {}   # nœud -> arête utilisée pour y arriver

    found = [None]

    def dfs2(u, prev):
        visited2.add(u)
        for v, edge in adj[u]:
            if found[0]:
                return
            if v == prev:
                continue
            if v in visited2:
                # Cycle trouvé entre u et v
                cycle = [edge]
                cur = u
                while cur != v:
                    cycle.append(edge_par[cur])
                    cur = parent2[cur]
                found[0] = cycle
                return
            parent2[v] = u
            edge_par[v] = edge
            dfs2(v, u)

    for start in range(n + m):
        if start not in visited2 and adj[start]:
            dfs2(start, -1)
            if found[0]:
                break

    if found[0]:
        # Convertir les arêtes (nœuds biparti) en paires (i,j) tableau
        cycle_ij = []
        for edge in found[0]:
            a, b = edge
            if a >= n:
                cycle_ij.append((b, a - n))
            else:
                cycle_ij.append((a, b - n))
        return cycle_ij
    return None


# ─────────────────────────────────────────────
# 6. GESTION DÉGÉNÉRESCENCE
# ─────────────────────────────────────────────

def nombre_de_base_requis(n, m):
    return n + m - 1


def corriger_degenere(n, m, alloc):
    """Ajoute des variables à zéro jusqu'à avoir n+m-1 arêtes de base (non connexe/dégénéré)."""
    requis = nombre_de_base_requis(n, m)
    aretes = _aretes(n, m, alloc)
    while len(aretes) < requis:
        # Cherche une case (i,j) hors base dont l'ajout ne crée pas de cycle
        ajout = False
        for i in range(n):
            for j in range(m):
                if alloc[i][j] is None:
                    alloc[i][j] = 0  # ajout provisoire
                    if not detecter_cycle(n, m, alloc):
                        print(f"  [Dégénérescence] Ajout de alloc[{i+1}][{j+1}]=0 pour compléter la base.")
                        ajout = True
                        break
                    else:
                        alloc[i][j] = None  # annuler
            if ajout:
                break
        if not ajout:
            break
        aretes = _aretes(n, m, alloc)

    # Supprimer les cycles éventuels
    cycle = detecter_cycle(n, m, alloc)
    if cycle:
        print(f"  [Cycle] Cycle détecté : {cycle} — suppression d'une arête à 0.")
        for (i, j) in cycle:
            if alloc[i][j] == 0:
                alloc[i][j] = None
                if not detecter_cycle(n, m, alloc):
                    break
                alloc[i][j] = 0  # remettre et essayer suivant


# ─────────────────────────────────────────────
# 7. MÉTHODE DES POTENTIELS (MODI)
# ─────────────────────────────────────────────

def calculer_potentiels(n, m, couts, alloc):
    """Calcule les potentiels E_S[i] et E_T[j] tels que
    E_S[i] + E_T[j] = c[i][j] pour toute arête de base.
    Résolution par propagation BFS.
    """
    potS = [None] * n
    potT = [None] * m
    potS[0] = 0  # convention : E_S[0] = 0

    aretes = _aretes(n, m, alloc)
    # BFS sur le graphe biparti des arêtes de base
    changed = True
    while changed:
        changed = False
        for (i, j) in aretes:
            ci = couts[i][j]
            if potS[i] is not None and potT[j] is None:
                potT[j] = ci - potS[i]
                changed = True
            elif potT[j] is not None and potS[i] is None:
                potS[i] = ci - potT[j]
                changed = True
    return potS, potT


def calculer_marginaux(n, m, couts, alloc, potS, potT):
    """d(i,j) = c(i,j) - (potS[i] + potT[j]) pour les cases hors base."""
    marginaux = [[None] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if alloc[i][j] is None:
                if potS[i] is not None and potT[j] is not None:
                    marginaux[i][j] = couts[i][j] - (potS[i] + potT[j])
    return marginaux


def trouver_arête_améliorante(n, m, marginaux):
    """Retourne (i,j) avec d(i,j) < 0 le plus négatif, ou None si optimal."""
    best = (None, None, 0)
    for i in range(n):
        for j in range(m):
            if marginaux[i][j] is not None and marginaux[i][j] < best[2]:
                best = (i, j, marginaux[i][j])
    if best[0] is None:
        return None
    return (best[0], best[1])


# ─────────────────────────────────────────────
# 8. CYCLE D'AMÉLIORATION (MARCHE-PIED)
# ─────────────────────────────────────────────

def trouver_cycle_stepping_stone(n, m, alloc, i0, j0):
    """Trouve le cycle élémentaire dans le graphe de base incluant (i0,j0).
    Le cycle alterne : case hors-base → ligne → colonne → ...
    On utilise un backtracking sur le graphe biparti.
    Retourne une liste ordonnée [(i,j), ...] de cases du cycle,
    commençant par (i0,j0), alternant + et -.
    """
    # Cases de base
    base = {(i, j) for i in range(n) for j in range(m) if alloc[i][j] is not None}
    base.add((i0, j0))  # on l'ajoute temporairement

    def voisins_ligne(i, j_exc):
        return [(i, jj) for (ii, jj) in base if ii == i and jj != j_exc]

    def voisins_col(j, i_exc):
        return [(ii, j) for (ii, jj) in base if jj == j and ii != i_exc]

    # DFS : chemin = liste de cases, alternance ligne/col
    result = [None]

    def dfs(path, mode):
        """mode='col': on cherche dans la même colonne que la dernière case."""
        if result[0]:
            return
        cur = path[-1]
        if mode == 'col':
            candidats = voisins_col(cur[1], cur[0])
        else:
            candidats = voisins_ligne(cur[0], cur[1])

        for nxt in candidats:
            if len(path) >= 4 and nxt == (i0, j0):
                result[0] = path[:]
                return
            if nxt != (i0, j0) and nxt not in path:
                path.append(nxt)
                dfs(path, 'ligne' if mode == 'col' else 'col')
                path.pop()
                if result[0]:
                    return

    dfs([(i0, j0)], 'col')
    base.discard((i0, j0))
    return result[0]


def ameliorer(alloc, cycle):
    """Applique le déplacement sur le cycle (+ impairs, - pairs).
    Retourne True si une amélioration a été faite.
    """
    # Indices pairs (+) et impairs (-)
    plus  = [cycle[k] for k in range(0, len(cycle), 2)]
    moins = [cycle[k] for k in range(1, len(cycle), 2)]

    theta = min(alloc[i][j] for (i, j) in moins if alloc[i][j] is not None)
    if theta <= 0:
        # Dégénéré mais on continue
        theta = 0

    for (i, j) in plus:
        if alloc[i][j] is None:
            alloc[i][j] = 0
        alloc[i][j] += theta

    sortantes = []
    for (i, j) in moins:
        alloc[i][j] -= theta
        if alloc[i][j] == 0:
            sortantes.append((i, j))

    # Retirer exactement une variable sortante (la première) pour garder n+m-1 variables
    # La case entrante (cycle[0]) entre dans la base
    if sortantes:
        i_out, j_out = sortantes[0]
        alloc[i_out][j_out] = None

    return True


# ─────────────────────────────────────────────
# 9. BOUCLE PRINCIPALE
# ─────────────────────────────────────────────

def resoudre(n, m, couts, provisions, demandes, methode='NO'):
    t0 = time.time()

    # Vérification équilibre
    if sum(provisions) != sum(demandes):
        print("ERREUR : problème non équilibré !")
        return

    afficher_matrice_couts(n, m, couts, provisions, demandes)

    # Solution initiale
    print(f"\n{'='*50}")
    print(f"Méthode initiale : {'Nord-Ouest' if methode=='NO' else 'Balas-Hammer (Vogel)'}")
    print('='*50)

    if methode == 'NO':
        alloc = nord_ouest(n, m, provisions, demandes)
    else:
        alloc = balas_hammer(n, m, couts, provisions, demandes)

    afficher_matrice_transport(n, m, couts, alloc, provisions, demandes)
    print(f"  Coût initial : {cout_total(n, m, couts, alloc)}")

    # Boucle d'optimisation
    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'─'*50}")
        print(f"Itération {iteration}")

        # Vérification et correction dégénérescence
        corriger_degenere(n, m, alloc)

        if not est_connexe(n, m, alloc):
            print("  [Graphe non connexe] Ajout d'arêtes fictives.")
            corriger_degenere(n, m, alloc)

        # Potentiels
        potS, potT = calculer_potentiels(n, m, couts, alloc)

        if None in potS or None in potT:
            print("  [Attention] Potentiels incomplets – vérifiez la connexité.")
            break

        afficher_potentiels_et_marginaux(n, m, couts, alloc, potS, potT)

        # Coûts marginaux
        marginaux = calculer_marginaux(n, m, couts, alloc, potS, potT)

        # Arête améliorante ?
        arête = trouver_arête_améliorante(n, m, marginaux)
        if arête is None:
            print("\n  ✓ Solution optimale atteinte.")
            break

        i0, j0 = arête
        print(f"\n  → Arête améliorante : ({i0+1},{j0+1}), d={marginaux[i0][j0]}")

        cycle = trouver_cycle_stepping_stone(n, m, alloc, i0, j0)
        if cycle is None:
            print("  [Erreur] Impossible de trouver le cycle d'amélioration.")
            break

        print(f"  → Cycle : {[(i+1,j+1) for i,j in cycle]}")
        ameliorer(alloc, cycle)
        afficher_matrice_transport(n, m, couts, alloc, provisions, demandes)
        print(f"  Coût après amélioration : {cout_total(n, m, couts, alloc)}")

    # Résultat final
    print(f"\n{'='*50}")
    print("SOLUTION OPTIMALE")
    print('='*50)
    afficher_matrice_transport(n, m, couts, alloc, provisions, demandes)
    print(f"\n  Coût total minimum : {cout_total(n, m, couts, alloc)}")
    print(f"  Temps d'exécution  : {time.time()-t0:.4f}s")


# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────

def main():
    # Choix du fichier
    if len(sys.argv) >= 2:
        chemin = sys.argv[1]
        n, m, couts, provisions, demandes = lire_fichier(chemin)
    else:
        choix = input("Fichier (f) ou génération aléatoire (a) ? [f/a] : ").strip().lower()
        if choix == 'a':
            n = int(input("Nombre d'origines n : "))
            m = int(input("Nombre de destinations m : "))
            n, m, couts, provisions, demandes = generer_aleatoire(n, m)
            print(f"Problème généré : n={n}, m={m}, ΣPi={sum(provisions)}, ΣCj={sum(demandes)}")
        else:
            chemin = input("Chemin du fichier .txt : ").strip()
            n, m, couts, provisions, demandes = lire_fichier(chemin)

    # Choix de la méthode initiale
    methode = input("\nMéthode initiale – Nord-Ouest (NO) ou Balas-Hammer (BH) ? [NO/BH] : ").strip().upper()
    if methode not in ('NO', 'BH'):
        methode = 'NO'

    resoudre(n, m, couts, provisions, demandes, methode)


if __name__ == '__main__':
    main()
