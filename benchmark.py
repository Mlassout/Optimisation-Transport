#!/usr/bin/env python3
"""
Benchmark de complexité — Problème de transport
Mesure θ (itérations MODI) et t (temps total) pour NO et BH.

Usage :
  python3 benchmark.py          → lance le benchmark complet puis trace les courbes
  python3 benchmark.py plot     → trace les courbes depuis le CSV existant (reprise)

Les résultats sont sauvegardés ligne par ligne dans CSV_FILE,
ce qui permet de reprendre si le programme est interrompu.
"""

import numpy as np
import time
import csv
import os
import sys
from collections import deque

import matplotlib
matplotlib.use('Agg')   # compatible headless (pas besoin d'écran)
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────────

N_VALUES   = [10, 40, 100, 400, 1000, 4000, 10000]
N_RUNS     = 100
CSV_FILE   = "resultats_benchmark.csv"
TIMEOUT_S  = 300        # secondes max par run individuel
MEM_LIMIT_MB = 4000     # skip si mémoire estimée > 4 GB


# ════════════════════════════════════════════
# 1. GÉNÉRATION ALÉATOIRE (numpy)
# ════════════════════════════════════════════

def gen_random(n):
    """Génère un problème de transport carré n×n équilibré."""
    couts = np.random.randint(1, 100, (n, n), dtype=np.int32)
    p = np.random.randint(50, 500, n, dtype=np.int64)
    d = np.random.randint(50, 500, n, dtype=np.int64)
    diff = int(p.sum() - d.sum())
    if diff > 0:
        d[-1] += diff
    elif diff < 0:
        p[-1] -= diff
    return couts, p, d


# ════════════════════════════════════════════
# 2. SOLUTIONS INITIALES
# ════════════════════════════════════════════

def nord_ouest_np(n, p, d):
    """Méthode du coin Nord-Ouest.
    alloc[i,j] = -1 → hors base, ≥0 → en base.
    """
    alloc = np.full((n, n), -1, dtype=np.int64)
    p, d = p.copy(), d.copy()
    i = j = 0
    while i < n and j < n:
        q = min(int(p[i]), int(d[j]))
        alloc[i, j] = q
        p[i] -= q
        d[j] -= q
        if p[i] == 0 and d[j] == 0:
            # dégénérescence : arête à 0 pour garder n+m-1 variables
            if i + 1 < n and j + 1 < n:
                alloc[i + 1, j] = 0
            i += 1
            j += 1
        elif p[i] == 0:
            i += 1
        else:
            j += 1
    return alloc


def balas_hammer_np(n, couts, p, d):
    """Méthode de Vogel (Balas-Hammer).
    Utilise np.partition (partial sort O(n)) pour les pénalités.
    """
    alloc = np.full((n, n), -1, dtype=np.int64)
    p, d = p.copy(), d.copy()
    act_r = list(range(n))
    act_c = list(range(n))

    while act_r and act_c:
        # Sous-matrice des lignes/colonnes encore actives
        sub = couts[np.ix_(act_r, act_c)]   # shape (|act_r|, |act_c|)
        nr, nc = len(act_r), len(act_c)

        # Pénalités lignes : 2e_min - min sur chaque ligne
        if nc >= 2:
            s2r = np.partition(sub, 1, axis=1)
            pen_r = (s2r[:, 1] - s2r[:, 0]).astype(np.int64)
        else:
            pen_r = sub[:, 0].astype(np.int64)

        # Pénalités colonnes : 2e_min - min sur chaque colonne
        if nr >= 2:
            s2c = np.partition(sub, 1, axis=0)
            pen_c = (s2c[1, :] - s2c[0, :]).astype(np.int64)
        else:
            pen_c = sub[0, :].astype(np.int64)

        mr = int(pen_r.max())
        mc = int(pen_c.max())

        if mr >= mc:
            ri = int(pen_r.argmax())
            i  = act_r[ri]
            ci = int(sub[ri].argmin())
            j  = act_c[ci]
        else:
            ci = int(pen_c.argmax())
            j  = act_c[ci]
            ri = int(sub[:, ci].argmin())
            i  = act_r[ri]

        q = int(min(p[i], d[j]))
        alloc[i, j] = q
        p[i] -= q
        d[j] -= q
        if p[i] == 0:
            act_r.remove(i)
        if d[j] == 0:
            act_c.remove(j)

    return alloc


# ════════════════════════════════════════════
# 3. GRAPHE / DÉGÉNÉRESCENCE
# ════════════════════════════════════════════

def get_basic(alloc):
    """Retourne la liste des cases de base sous forme [(i,j), ...]."""
    rows, cols = np.where(alloc >= 0)
    return list(zip(rows.tolist(), cols.tolist()))


def build_adj(n, basic):
    """Adjacence du graphe biparti pour les cases de base.
    row_cols[i] = liste des j tels que (i,j) ∈ base.
    col_rows[j] = liste des i tels que (i,j) ∈ base.
    """
    row_cols = [[] for _ in range(n)]
    col_rows = [[] for _ in range(n)]
    for (i, j) in basic:
        row_cols[i].append(j)
        col_rows[j].append(i)
    return row_cols, col_rows


def corriger_degenere_np(n, alloc):
    """Ajoute des 0 hors-base jusqu'à avoir exactement 2n-1 variables de base,
    en utilisant Union-Find pour garantir l'absence de cycle.
    """
    requis = 2 * n - 1
    basic  = get_basic(alloc)
    if len(basic) >= requis:
        return

    # Union-Find sur les nœuds biparti : sources 0..n-1, destinations n..2n-1
    parent = list(range(2 * n))
    rank   = [0] * (2 * n)

    def find(x):
        # Path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False   # cycle → refuser
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    # Initialiser UF avec les arêtes existantes
    for (i, j) in basic:
        union(i, n + j)

    # Ajouter des arêtes sans créer de cycle
    for i in range(n):
        for j in range(n):
            if len(basic) >= requis:
                return
            if alloc[i, j] < 0 and union(i, n + j):
                alloc[i, j] = 0
                basic.append((i, j))


# ════════════════════════════════════════════
# 4. POTENTIELS (BFS)
# ════════════════════════════════════════════

def calculer_potentiels_np(n, couts, basic):
    """Calcule E_S[i] et E_T[j] tels que E_S[i] + E_T[j] = c[i,j] ∀ (i,j) ∈ base.
    BFS sur le graphe biparti des arêtes de base (2n-1 nœuds à visiter).
    """
    row_cols, col_rows = build_adj(n, basic)
    potS = [None] * n
    potT = [None] * n
    potS[0] = 0  # convention : E_S[0] = 0

    q = deque([('s', 0)])
    while q:
        typ, node = q.popleft()
        if typ == 's':
            i = node
            for j in row_cols[i]:
                if potT[j] is None:
                    potT[j] = int(couts[i, j]) - potS[i]
                    q.append(('t', j))
        else:
            j = node
            for i in col_rows[j]:
                if potS[i] is None:
                    potS[i] = int(couts[i, j]) - potT[j]
                    q.append(('s', i))

    return potS, potT


# ════════════════════════════════════════════
# 5. CYCLE MARCHE-PIED (backtracking + adjacence)
# ════════════════════════════════════════════

def trouver_cycle_np(n, basic, i0, j0):
    """Trouve le cycle élémentaire passant par la case entrante (i0,j0).
    Backtracking DFS sur le graphe biparti étendu (base + (i0,j0)).
    Alterne : cherche dans la même colonne, puis même ligne, etc.
    Utilise un set path_set pour éviter O(n) de recherche dans path.
    """
    ext = basic + [(i0, j0)]
    row_cols, col_rows = build_adj(n, ext)

    result   = [None]
    path_set = set()   # éléments intermédiaires du chemin courant

    def dfs(path, mode):
        if result[0]:
            return
        ci, cj = path[-1]

        if mode == 'col':
            neighbors = [(ii, cj) for ii in col_rows[cj] if ii != ci]
        else:
            neighbors = [(ci, jj) for jj in row_cols[ci] if jj != cj]

        for nxt in neighbors:
            # Fermeture du cycle : on revient au point de départ
            if len(path) >= 4 and nxt == (i0, j0):
                result[0] = path[:]
                return
            # Avancer sans revisiter
            if nxt != (i0, j0) and nxt not in path_set:
                path.append(nxt)
                path_set.add(nxt)
                dfs(path, 'ligne' if mode == 'col' else 'col')
                path.pop()
                path_set.discard(nxt)
                if result[0]:
                    return

    dfs([(i0, j0)], 'col')
    return result[0]


def ameliorer_np(alloc, cycle):
    """Applique le déplacement θ sur le cycle (cases + aux rangs pairs, - aux rangs impairs)."""
    plus  = cycle[0::2]
    moins = cycle[1::2]

    theta = min(int(alloc[i, j]) for (i, j) in moins)

    for (i, j) in plus:
        if alloc[i, j] < 0:
            alloc[i, j] = 0
        alloc[i, j] += theta

    sortantes = []
    for (i, j) in moins:
        alloc[i, j] -= theta
        if alloc[i, j] == 0:
            sortantes.append((i, j))

    # Retirer une variable sortante (la première) : case qui vaut 0 parmi les moins
    if sortantes:
        alloc[sortantes[0][0], sortantes[0][1]] = -1


# ════════════════════════════════════════════
# 6. RÉSOLUTION COMPLÈTE
# ════════════════════════════════════════════

def resoudre_np(n, couts, p, d, methode):
    """Résout le problème et retourne (nb_iterations, elapsed_s).
    Retourne (None, None) si le timeout est dépassé.

    Optimisation clé : le calcul des coûts marginaux est vectorisé :
        marg = couts - (potS[:,None] + potT[None,:])
    → une seule opération numpy sur la matrice n×n au lieu d'une double boucle Python.
    """
    t0 = time.time()

    if methode == 'NO':
        alloc = nord_ouest_np(n, p, d)
    else:
        alloc = balas_hammer_np(n, couts, p, d)

    nb_iter = 0
    LIMIT   = 20 * n   # borne de sécurité anti-boucle infinie

    for _ in range(LIMIT):
        if time.time() - t0 > TIMEOUT_S:
            return None, None

        corriger_degenere_np(n, alloc)

        basic = get_basic(alloc)
        potS, potT = calculer_potentiels_np(n, couts, basic)

        if None in potS or None in potT:
            break  # graphe non connexe (ne devrait pas arriver)

        # ── Coûts marginaux vectorisés (numpy broadcasting) ──
        pS   = np.array(potS, dtype=np.float64)
        pT   = np.array(potT, dtype=np.float64)
        marg = couts.astype(np.float64) - (pS[:, None] + pT[None, :])

        # Masquer les cases de base : on les met à +inf
        marg_hb = np.where(alloc < 0, marg, np.inf)

        min_val = float(marg_hb.min())
        if min_val >= -1e-9:
            break   # solution optimale

        # Case entrante : le coût marginal le plus négatif
        i0, j0 = np.unravel_index(marg_hb.argmin(), (n, n))
        i0, j0 = int(i0), int(j0)

        cycle = trouver_cycle_np(n, basic, i0, j0)
        if cycle is None:
            break

        ameliorer_np(alloc, cycle)
        nb_iter += 1

    return nb_iter, time.time() - t0


# ════════════════════════════════════════════
# 7. BENCHMARK PRINCIPAL
# ════════════════════════════════════════════

def run_benchmark():
    """Lance les N_RUNS résolutions pour chaque n ∈ N_VALUES.
    Sauvegarde ligne par ligne dans CSV_FILE (reprise possible).
    """
    # Reprise si CSV existant
    done = set()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, newline='') as f:
            for row in csv.DictReader(f):
                done.add((int(row['n']), int(row['run'])))
        if done:
            print(f"  Reprise depuis '{CSV_FILE}' ({len(done)} runs déjà effectués).")

    fieldnames = ['n', 'run', 'theta_NO', 'theta_BH', 't_NO', 't_BH']
    mode = 'a' if done else 'w'

    with open(CSV_FILE, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not done:
            writer.writeheader()

        for n in N_VALUES:
            # Estimation mémoire : couts (int32) + alloc (int64) + marg (float64)
            mem_mb = (n * n * (4 + 8 + 8)) / 1e6
            print(f"\n{'─'*50}")
            print(f"  n = {n:<6}  |  mémoire estimée ≈ {mem_mb:.0f} MB")
            print('─'*50)

            if mem_mb > MEM_LIMIT_MB:
                print(f"  → Skippé (mémoire estimée dépasse {MEM_LIMIT_MB} MB)")
                continue

            t_n      = time.time()
            nb_skip  = 0

            for run in range(N_RUNS):
                if (n, run) in done:
                    continue

                couts, p, d = gen_random(n)

                th_NO, t_NO = resoudre_np(n, couts, p.copy(), d.copy(), 'NO')
                th_BH, t_BH = resoudre_np(n, couts, p.copy(), d.copy(), 'BH')

                if th_NO is None or th_BH is None:
                    nb_skip += 1
                    # Valeur négative = timeout, ignorée dans les graphiques
                    th_NO = th_NO if th_NO is not None else -1
                    th_BH = th_BH if th_BH is not None else -1
                    t_NO  = t_NO  if t_NO  is not None else TIMEOUT_S
                    t_BH  = t_BH  if t_BH  is not None else TIMEOUT_S

                writer.writerow({
                    'n': n, 'run': run,
                    'theta_NO': th_NO, 'theta_BH': th_BH,
                    't_NO': t_NO,      't_BH': t_BH,
                })
                f.flush()

                if (run + 1) % 10 == 0:
                    elapsed = time.time() - t_n
                    pace    = elapsed / (run + 1)
                    eta     = pace * (N_RUNS - run - 1)
                    print(f"  run {run+1:>3}/{N_RUNS}  |  {pace:.3f} s/run  |  ETA {eta:.0f} s")

            total_n = time.time() - t_n
            print(f"  n={n} terminé en {total_n:.1f}s  (timeouts: {nb_skip}/{N_RUNS})")


# ════════════════════════════════════════════
# 8. GRAPHIQUES (6 nuages de points)
# ════════════════════════════════════════════

def plot_resultats():
    """Trace les 6 nuages de points demandés."""
    if not os.path.exists(CSV_FILE):
        print("Pas de données. Lancez d'abord le benchmark.")
        return

    # Chargement CSV
    data = {}
    with open(CSV_FILE, newline='') as f:
        for row in csv.DictReader(f):
            n = int(row['n'])
            if n not in data:
                data[n] = {k: [] for k in ('theta_NO', 'theta_BH', 't_NO', 't_BH')}
            for k in ('theta_NO', 'theta_BH', 't_NO', 't_BH'):
                v = float(row[k])
                if v >= 0:   # ignorer les timeouts (valeur = -1)
                    data[n][k].append(v)

    if not data:
        print("CSV vide ou toutes les valeurs sont des timeouts.")
        return

    ns = sorted(data.keys())

    # Configuration des 6 sous-graphiques
    plots_cfg = [
        # (clé(s), label y, titre, couleur)
        ('theta_NO',
         r'$\theta_{NO}(n)$',
         r'Itérations MODI — départ Nord-Ouest',
         'steelblue'),
        ('theta_BH',
         r'$\theta_{BH}(n)$',
         r'Itérations MODI — départ Balas-Hammer',
         'darkorange'),
        ('t_NO',
         r'$t_{NO}(n)$ [s]',
         r'Temps total — Nord-Ouest + MODI',
         'steelblue'),
        ('t_BH',
         r'$t_{BH}(n)$ [s]',
         r'Temps total — Balas-Hammer + MODI',
         'darkorange'),
        (('theta_NO', 't_NO'),
         r'$\theta_{NO} + t_{NO}$',
         r'$\theta_{NO}(n) + t_{NO}(n)$',
         'steelblue'),
        (('theta_BH', 't_BH'),
         r'$\theta_{BH} + t_{BH}$',
         r'$\theta_{BH}(n) + t_{BH}(n)$',
         'darkorange'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(
        f'Complexité — Problème de transport ({N_RUNS} runs/n, problèmes carrés n×n)',
        fontsize=12, fontweight='bold'
    )

    rng = np.random.default_rng(42)

    for ax, (key, ylabel, title, color) in zip(axes.flat, plots_cfg):
        for n in ns:
            if isinstance(key, tuple):
                a = np.array(data[n][key[0]])
                b = np.array(data[n][key[1]])
                if len(a) == 0 or len(b) == 0:
                    continue
                vals = a[:min(len(a), len(b))] + b[:min(len(a), len(b))]
            else:
                vals = np.array(data[n][key])

            if len(vals) == 0:
                continue

            # Léger jitter horizontal pour séparer les points superposés
            jitter = rng.uniform(-0.02, 0.02, len(vals)) * n
            ax.scatter(
                np.full(len(vals), n, dtype=float) + jitter,
                vals,
                alpha=0.30, s=7, color=color, linewidths=0,
            )

            # Médiane (point noir contouré)
            ax.scatter(
                [n], [float(np.median(vals))],
                s=55, color=color, edgecolors='black', linewidths=0.8, zorder=5,
            )

        ax.set_xscale('log')
        # Log y seulement si toutes les valeurs sont > 0
        all_vals = []
        for n in ns:
            if isinstance(key, tuple):
                a = data[n].get(key[0], [])
                b = data[n].get(key[1], [])
                k_ = min(len(a), len(b))
                all_vals += [x + y for x, y in zip(a[:k_], b[:k_])]
            else:
                all_vals += data[n].get(key, [])
        if all_vals and min(all_vals) > 0:
            ax.set_yscale('log')

        ax.set_xlabel('n', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns], rotation=40, fontsize=7)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    out_path = 'resultats_benchmark.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nGraphique sauvegardé : {out_path}")


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        plot_resultats()
    else:
        print("╔══════════════════════════════════════════════╗")
        print("║  Benchmark — Problème de transport           ║")
        print(f"║  n ∈ {N_VALUES}  ║")
        print(f"║  {N_RUNS} runs/n  |  timeout = {TIMEOUT_S}s/run         ║")
        print("╚══════════════════════════════════════════════╝")
        print(f"\nRésultats → {CSV_FILE}")
        print("(interruptible : relancer pour reprendre)\n")

        t0 = time.time()
        run_benchmark()
        elapsed = time.time() - t0
        print(f"\nBenchmark terminé en {elapsed:.1f}s  ({elapsed/60:.1f} min)")

        plot_resultats()
