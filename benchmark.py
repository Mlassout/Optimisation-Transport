#!/usr/bin/env python3
"""
Benchmark de complexité — Problème de transport  (§3.3 des consignes)

Définitions (conformes au sujet) :
  θNO(n)  = temps d'exécution de l'algorithme Nord-Ouest seul
  θBH(n)  = temps d'exécution de l'algorithme Balas-Hammer seul
  tNO(n)  = temps du marche-pied avec potentiel (départ Nord-Ouest)
  tBH(n)  = temps du marche-pied avec potentiel (départ Balas-Hammer)

Génération (§3.3.1) :
  temp[i][j] ∈ [1,100]  →  P_i = Σ_j temp[i][j],  C_j = Σ_i temp[i][j]
  Garantit l'équilibre Σ P_i = Σ C_j sans correction.

Usage :
  python3 benchmark.py        → benchmark complet + graphiques
  python3 benchmark.py plot   → graphiques depuis le CSV existant
"""

import numpy as np
import time
import csv
import os
import sys
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────────

N_VALUES     = [10, 40, 100, 400, 1000, 4000, 10000]
N_RUNS       = 100
CSV_FILE     = "testLiam.csv"
TIMEOUT_S    = 300      # max secondes pour le marche-pied par run
MEM_LIMIT_MB = 4000     # skip si estimation mémoire > 4 GB


# ════════════════════════════════════════════
# 1. GÉNÉRATION (§3.3.1)
# ════════════════════════════════════════════

def gen_random(n):
    """Génère un problème carré n×n équilibré selon les consignes :
    - a[i,j]    ∈ [1,100]   entier (coûts)
    - temp[i,j] ∈ [1,100]   entier
    - P_i = Σ_j temp[i,j]   (somme de la ligne i de temp)
    - C_j = Σ_i temp[i,j]   (somme de la colonne j de temp)
    Équilibre automatique : Σ P_i = Σ C_j = Σ_ij temp[i,j].
    """
    couts = np.random.randint(1, 101, (n, n), dtype=np.int32)
    temp  = np.random.randint(1, 101, (n, n), dtype=np.int64)
    p     = temp.sum(axis=1)   # provisions
    d     = temp.sum(axis=0)   # demandes
    return couts, p, d


# ════════════════════════════════════════════
# 2. SOLUTIONS INITIALES
# ════════════════════════════════════════════

def nord_ouest_np(n, p, d):
    """Méthode du coin Nord-Ouest.
    alloc[i,j] = -1 → hors base  /  ≥ 0 → en base.
    """
    alloc = np.full((n, n), -1, dtype=np.int64)
    p, d  = p.copy(), d.copy()
    i = j = 0
    while i < n and j < n:
        q = int(min(p[i], d[j]))
        alloc[i, j] = q
        p[i] -= q
        d[j] -= q
        if p[i] == 0 and d[j] == 0:
            if i + 1 < n and j + 1 < n:
                alloc[i + 1, j] = 0   # dégénérescence : arête fictive à 0
            i += 1
            j += 1
        elif p[i] == 0:
            i += 1
        else:
            j += 1
    return alloc


def balas_hammer_np(n, couts, p, d):
    """Méthode de Vogel (Balas-Hammer).
    np.partition(sub, 1, axis=…) donne le 2e minimum sans trier complètement.
    """
    alloc = np.full((n, n), -1, dtype=np.int64)
    p, d  = p.copy(), d.copy()
    act_r = list(range(n))
    act_c = list(range(n))

    while act_r and act_c:
        sub    = couts[np.ix_(act_r, act_c)]
        nr, nc = len(act_r), len(act_c)

        if nc >= 2:
            s2r   = np.partition(sub, 1, axis=1)
            pen_r = (s2r[:, 1] - s2r[:, 0]).astype(np.int64)
        else:
            pen_r = sub[:, 0].astype(np.int64)

        if nr >= 2:
            s2c   = np.partition(sub, 1, axis=0)
            pen_c = (s2c[1, :] - s2c[0, :]).astype(np.int64)
        else:
            pen_c = sub[0, :].astype(np.int64)

        mr = int(pen_r.max())
        mc = int(pen_c.max())

        if mr >= mc:
            ri = int(pen_r.argmax());  i = act_r[ri]
            ci = int(sub[ri].argmin()); j = act_c[ci]
        else:
            ci = int(pen_c.argmax());  j = act_c[ci]
            ri = int(sub[:, ci].argmin()); i = act_r[ri]

        q = int(min(p[i], d[j]))
        alloc[i, j] = q
        p[i] -= q
        d[j] -= q
        if p[i] == 0: act_r.remove(i)
        if d[j] == 0: act_c.remove(j)

    return alloc


# ════════════════════════════════════════════
# 3. GRAPHE / DÉGÉNÉRESCENCE (Union-Find)
# ════════════════════════════════════════════

def get_basic(alloc):
    rows, cols = np.where(alloc >= 0)
    return list(zip(rows.tolist(), cols.tolist()))


def build_adj(n, basic):
    row_cols = [[] for _ in range(n)]
    col_rows = [[] for _ in range(n)]
    for (i, j) in basic:
        row_cols[i].append(j)
        col_rows[j].append(i)
    return row_cols, col_rows


def corriger_degenere_np(n, alloc):
    """Complète la base à 2n-1 arêtes sans créer de cycle (Union-Find)."""
    requis = 2 * n - 1
    basic  = get_basic(alloc)
    if len(basic) >= requis:
        return

    parent = list(range(2 * n))
    rank   = [0] * (2 * n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    for (i, j) in basic:
        union(i, n + j)

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
    row_cols, col_rows = build_adj(n, basic)
    potS = [None] * n
    potT = [None] * n
    potS[0] = 0

    q = deque([('s', 0)])
    while q:
        typ, node = q.popleft()
        if typ == 's':
            for j in row_cols[node]:
                if potT[j] is None:
                    potT[j] = int(couts[node, j]) - potS[node]
                    q.append(('t', j))
        else:
            for i in col_rows[node]:
                if potS[i] is None:
                    potS[i] = int(couts[i, node]) - potT[node]
                    q.append(('s', i))

    return potS, potT


# ════════════════════════════════════════════
# 5. CYCLE MARCHE-PIED
# ════════════════════════════════════════════

def trouver_cycle_np(n, basic, i0, j0):
    ext      = basic + [(i0, j0)]
    row_cols, col_rows = build_adj(n, ext)
    from collections import deque
    queue = deque([[(i0, j0)]])

    while queue:
        path = queue.popleft()
        ci, cj = path[-1]

        mode = 'col' if len(path) % 2 == 1 else 'ligne'

        neighbors = (
            [(ii, cj) for ii in col_rows[cj] if ii != ci]
            if mode == 'col'
            else [(ci, jj) for jj in row_cols[ci] if jj != cj]
        )

        for nxt in neighbors:
            if len(path) >= 4 and nxt == (i0, j0):
                return path

            if nxt not in path:
                queue.append(path + [nxt])

    return None
    #result   = [None]

"""
    visited = set()

    def dfs(path, mode):
        if result[0]:
            return
        if len(path) > 2 * n:
            return

        ci, cj = path[-1]
        state = (ci, cj, mode)
        if state in visited:
            return
        visited.add(state)
        neighbors = ([(ii, cj) for ii in col_rows[cj] if ii != ci]
                     if mode == 'col'
                     else [(ci, jj) for jj in row_cols[ci] if jj != cj])
        if not neighbors:
            return

        for nxt in neighbors:
            if len(path) >= 2 and nxt == path[-2]:
                continue
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
    return result[0]

"""
def ameliorer_np(alloc, cycle):
    plus  = cycle[0::2]
    moins = cycle[1::2]
    theta = min(int(alloc[i, j]) for (i, j) in moins)
    for (i, j) in plus:
        if alloc[i, j] < 0:
            alloc[i, j] = 0
        alloc[i, j] += theta
    sortantes = [(i, j) for (i, j) in moins
                 if int(alloc[i, j]) - theta == 0]
    for (i, j) in moins:
        alloc[i, j] -= theta
    if sortantes:
        alloc[sortantes[0][0], sortantes[0][1]] = -1


# ════════════════════════════════════════════
# 6. RÉSOLUTION  →  retourne (θ, t)
# ════════════════════════════════════════════

def resoudre_np(n, couts, p, d, methode):
    """Retourne (theta, t_modi) en secondes (time.perf_counter).
    theta   = temps de l'initialisation NO ou BH
    t_modi  = temps du marche-pied avec potentiel
    Retourne (None, None) si le marche-pied dépasse TIMEOUT_S.
    """
    # ── Initialisation ──────────────────────────────────────
    t0 = time.perf_counter()
    if methode == 'NO':
        alloc = nord_ouest_np(n, p, d)
    else:
        alloc = balas_hammer_np(n, couts, p, d)
    theta = time.perf_counter() - t0

    # ── Marche-pied avec potentiel ───────────────────────────
    t1    = time.perf_counter()
    LIMIT = 20 * n   # borne de sécurité anti-boucle

    for _ in range(LIMIT):
        if time.perf_counter() - t1 > TIMEOUT_S:
            return None, None

        corriger_degenere_np(n, alloc)

        basic = get_basic(alloc)
        potS, potT = calculer_potentiels_np(n, couts, basic)
        if None in potS or None in potT:
            break

        # Coûts marginaux — calcul vectorisé (clé de performance)
        pS   = np.array(potS, dtype=np.float64)
        pT   = np.array(potT, dtype=np.float64)
        marg = couts.astype(np.float64) - (pS[:, None] + pT[None, :])
        marg_hb = np.where(alloc < 0, marg, np.inf)

        if float(marg_hb.min()) >= -1e-9:
            break   # optimal

        i0, j0 = np.unravel_index(marg_hb.argmin(), (n, n))
        cycle  = trouver_cycle_np(n, basic, int(i0), int(j0))
        if cycle is None:
            break
        ameliorer_np(alloc, cycle)

    t_modi = time.perf_counter() - t1
    return theta, t_modi


# ════════════════════════════════════════════
# 7. BENCHMARK
# ════════════════════════════════════════════

def run_benchmark():
    """100 runs × 7 valeurs de n.  Sauvegarde CSV ligne par ligne (reprise OK)."""
    done = set()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, newline='') as f:
            for row in csv.DictReader(f):
                done.add((int(row['n']), int(row['run'])))
        if done:
            print(f"  Reprise depuis '{CSV_FILE}'  ({len(done)} runs déjà faits).")

    fieldnames = ['n', 'run', 'theta_NO', 'theta_BH', 't_NO', 't_BH']
    mode = 'a' if done else 'w'

    with open(CSV_FILE, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not done:
            writer.writeheader()

        for n in N_VALUES:
            mem_mb = n * n * (4 + 8 + 8) / 1e6
            print(f"\n{'─'*52}")
            print(f"  n = {n:<6}  |  mémoire estimée ≈ {mem_mb:.0f} MB")
            print('─'*52)

            if mem_mb > MEM_LIMIT_MB:
                print(f"  → Skippé (>{MEM_LIMIT_MB} MB)")
                continue

            t_n = time.perf_counter()
            nb_timeout = 0

            for run in range(N_RUNS):
                if (n, run) in done:
                    continue

                couts, p, d = gen_random(n)

                th_NO, t_NO = resoudre_np(n, couts, p.copy(), d.copy(), 'NO')
                th_BH, t_BH = resoudre_np(n, couts, p.copy(), d.copy(), 'BH')

                if th_NO is None or th_BH is None:
                    nb_timeout += 1
                    th_NO = th_NO if th_NO is not None else -1.0
                    th_BH = th_BH if th_BH is not None else -1.0
                    t_NO  = t_NO  if t_NO  is not None else float(TIMEOUT_S)
                    t_BH  = t_BH  if t_BH  is not None else float(TIMEOUT_S)

                writer.writerow({'n': n, 'run': run,
                                 'theta_NO': th_NO, 'theta_BH': th_BH,
                                 't_NO': t_NO,      't_BH': t_BH})
                f.flush()

                if (run + 1) % 10 == 0:
                    elapsed = time.perf_counter() - t_n
                    pace    = elapsed / (run + 1)
                    eta     = pace * (N_RUNS - run - 1)
                    print(f"  run {run+1:>3}/{N_RUNS}  |  {pace:.3f} s/run  |  ETA {eta:.0f} s")

            print(f"  n={n} terminé en {time.perf_counter()-t_n:.1f}s  "
                  f"(timeouts : {nb_timeout}/{N_RUNS})")


# ════════════════════════════════════════════
# 8. GRAPHIQUES
# ════════════════════════════════════════════

def _charger_csv():
    data = {}
    with open(CSV_FILE, newline='') as f:
        for row in csv.DictReader(f):
            n = int(row['n'])
            if n not in data:
                data[n] = {k: [] for k in ('theta_NO', 'theta_BH', 't_NO', 't_BH')}
            for k in ('theta_NO', 'theta_BH', 't_NO', 't_BH'):
                v = float(row[k])
                if v >= 0:
                    data[n][k].append(v)
    return data


def plot_resultats():
    """Trace les 6 nuages de points + enveloppe supérieure + graphique du ratio."""
    if not os.path.exists(CSV_FILE):
        print("Pas de données.  Lancez d'abord le benchmark.")
        return

    data = _charger_csv()
    if not data:
        print("CSV vide.")
        return

    ns = sorted(data.keys())
    rng = np.random.default_rng(42)

    # ── Figure 1 : 6 nuages de points + enveloppe supérieure ──────────────
    plots_cfg = [
        ('theta_NO',           r'$\theta_{NO}(n)$  [s]',  'Temps init. Nord-Ouest',       'steelblue'),
        ('theta_BH',           r'$\theta_{BH}(n)$  [s]',  'Temps init. Balas-Hammer',     'darkorange'),
        ('t_NO',               r'$t_{NO}(n)$  [s]',       'Temps marche-pied (départ NO)', 'steelblue'),
        ('t_BH',               r'$t_{BH}(n)$  [s]',       'Temps marche-pied (départ BH)','darkorange'),
        (('theta_NO', 't_NO'), r'$\theta_{NO}+t_{NO}$  [s]', r'$\theta_{NO}+t_{NO}$ (total NO)', 'steelblue'),
        (('theta_BH', 't_BH'), r'$\theta_{BH}+t_{BH}$  [s]', r'$\theta_{BH}+t_{BH}$ (total BH)', 'darkorange'),
    ]

    fig1, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig1.suptitle(
        f'Complexité — Problème de transport  ({N_RUNS} runs/n, matrice n×n)',
        fontsize=12, fontweight='bold'
    )

    for ax, (key, ylabel, title, color) in zip(axes.flat, plots_cfg):
        maxs = []
        for n in ns:
            vals = _get_vals(data, n, key)
            if len(vals) == 0:
                maxs.append(np.nan)
                continue
            # Nuage (jitter léger)
            jitter = rng.uniform(-0.015, 0.015) * n
            ax.scatter(np.full(len(vals), n) + jitter, vals,
                       alpha=0.25, s=7, color=color, linewidths=0)
            maxs.append(float(np.max(vals)))

        # Enveloppe supérieure
        ns_ok  = [n for n, m in zip(ns, maxs) if not np.isnan(m)]
        maxs_ok = [m for m in maxs if not np.isnan(m)]
        if ns_ok:
            ax.plot(ns_ok, maxs_ok, color='black', linewidth=1.5,
                    marker='D', markersize=5, label='max (pire cas)', zorder=6)
            ax.legend(fontsize=8)

        _format_ax(ax, ns, ylabel, title)

    plt.tight_layout()
    out1 = 'benchmark_nuages.png'
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"Graphique 1 sauvegardé : {out1}")

    # ── Figure 2 : ratio (θNO+tNO) / (θBH+tBH) ───────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    fig2.suptitle(r'Ratio $(\theta_{NO}+t_{NO})\,/\,(\theta_{BH}+t_{BH})$ en fonction de $n$',
                  fontsize=12, fontweight='bold')

    ratio_maxs = []
    for n in ns:
        no_vals = _get_vals(data, n, ('theta_NO', 't_NO'))
        bh_vals = _get_vals(data, n, ('theta_BH', 't_BH'))
        k = min(len(no_vals), len(bh_vals))
        if k == 0:
            ratio_maxs.append(np.nan)
            continue
        ratios = no_vals[:k] / np.where(bh_vals[:k] > 0, bh_vals[:k], np.nan)
        ratios  = ratios[~np.isnan(ratios)]
        jitter  = rng.uniform(-0.015, 0.015) * n
        ax2.scatter(np.full(len(ratios), n) + jitter, ratios,
                    alpha=0.3, s=8, color='mediumseagreen', linewidths=0)
        ratio_maxs.append(float(np.max(ratios)))

    ns_ok    = [n for n, m in zip(ns, ratio_maxs) if not np.isnan(m)]
    maxs_ok  = [m for m in ratio_maxs if not np.isnan(m)]
    if ns_ok:
        ax2.plot(ns_ok, maxs_ok, color='black', linewidth=1.5,
                 marker='D', markersize=5, label='max (pire cas)', zorder=6)
        ax2.axhline(1.0, color='red', linestyle='--', linewidth=1, label='ratio = 1')
        ax2.legend(fontsize=9)

    _format_ax(ax2, ns,
               r'$(\theta_{NO}+t_{NO})\,/\,(\theta_{BH}+t_{BH})$',
               'Ratio total NO / total BH')

    plt.tight_layout()
    out2 = 'benchmark_ratio.png'
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Graphique 2 sauvegardé : {out2}")


def _get_vals(data, n, key):
    """Retourne un array numpy des valeurs pour une métrique (simple ou somme)."""
    if isinstance(key, tuple):
        a = np.array(data[n].get(key[0], []))
        b = np.array(data[n].get(key[1], []))
        k = min(len(a), len(b))
        return (a[:k] + b[:k]) if k > 0 else np.array([])
    return np.array(data[n].get(key, []))


def _format_ax(ax, ns, ylabel, title):
    """Mise en forme commune des axes."""
    ax.set_xscale('log')
    ylims = ax.get_ylim()
    if ylims[0] > 0:
        ax.set_yscale('log')
    ax.set_xlabel('n', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], rotation=40, fontsize=7)
    ax.grid(True, alpha=0.3, linestyle='--')


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        plot_resultats()
    else:
        print("╔══════════════════════════════════════════════════╗")
        print("║   Benchmark complexité — Problème de transport   ║")
        print(f"║   n ∈ {N_VALUES}   ║")
        print(f"║   {N_RUNS} runs/n  |  timeout marche-pied = {TIMEOUT_S}s    ║")
        print("╚══════════════════════════════════════════════════╝")
        print(f"  θ = temps init (NO ou BH),  t = temps marche-pied")
        print(f"  Résultats → {CSV_FILE}  (reprise automatique si interrompu)\n")

        t0 = time.perf_counter()
        run_benchmark()
        elapsed = time.perf_counter() - t0
        print(f"\nBenchmark terminé en {elapsed:.1f}s  ({elapsed/60:.1f} min)")
        plot_resultats()
