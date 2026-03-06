"""
Ablation study for Dri-MED and Dri-IMED on abrupt drift.

Two boxplot figures of final cumulative regret saved in figures/:
  1. figures/ablation_quantile.{png,pdf}
     baseline policy quantile in {0.25, 0.50, 0.75}  (epsilon fixed)
  2. figures/ablation_epsilon.{png,pdf}
     satisficing tolerance epsilon in {0.05, 0.10, 0.20, 0.30, 0.50}  (quantile=0.5 fixed)
"""

import argparse
import inspect
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from joblib import Parallel, delayed

from environment import Environment, DriftFunctions
from dri_imed    import DRI_IMED
from dri_med     import DRI_MED

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared style — mirrors run_experiments.py
# ---------------------------------------------------------------------------

COLORS = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
          '#a65628', '#984ea3', '#999999', '#e41a1c']

ALGO_NAMES  = ["Dri-MED", "Dri-IMED"]
ALGO_COLORS = {"Dri-MED": "#a65628", "Dri-IMED": "#984ea3"}

plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linestyle":     ":",
    "grid.alpha":         0.5,
    "axes.axisbelow":     True,
})

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

DRIFT_FN = DriftFunctions.abrupt(100)

ENV_CFG = dict(
    feature_dim      = 4,
    theta_norm       = 1.0,
    num_actions      = 5,
    num_users        = 10,
    num_episodes     = 1000,
    feedback_dim     = 4,
    preference_scale = 2.0,
    context_dim      = 2,
    context_noise    = 1.,
    reward_noise     = 0.1,
)

DELTA             = 0.01
d                 = ENV_CFG["feature_dim"]
S                 = ENV_CFG["theta_norm"]
GAMMA_DECAY       = 0.99

DEFAULT_QUANTILE  = 0.50
DEFAULT_EPSILON   = 0.10

QUANTILE_GRID  = [0.25, 0.50, 0.75]
EPSILON_GRID   = [0.05, 0.10, 0.20, 0.30, 0.50]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one_seed(algo_cls, algo_kwargs, pi0, seed):
    env      = Environment(**ENV_CFG, drift_fn=DRIFT_FN, seed=42)
    env._rng = np.random.default_rng(seed)

    H = env.num_users
    L = env.num_episodes

    kw = dict(algo_kwargs)
    if "pi0" in inspect.signature(algo_cls.__init__).parameters:
        kw["pi0"] = pi0
    algo = algo_cls(**kw)

    opt_arms   = env.optimal_policy()
    rng        = np.random.default_rng(seed)
    reg_acc    = 0.0
    dummy_arms = np.zeros(H, dtype=int)

    for ell in range(L):
        prefs      = env.get_preferences()
        ctx        = env.sample_contexts(ell)
        features   = env.compute_features(prefs)
        sigma_mat  = env.compute_sigma_matrix(ctx, prefs)
        true_means = features @ env.theta

        algo.start_episode(features, sigma_mat)
        for h in range(H):
            arm_h         = algo.select_arm(features[h], sigma_mat[h], rng)
            dummy_arms[h] = arm_h
            opt_a         = int(opt_arms[h])
            reg_acc      += max(0.0,
                                float(true_means[h, opt_a]) -
                                float(true_means[h, arm_h]))
            r_all_h, _    = env.compute_rewards(dummy_arms, prefs, ctx)
            algo.update(features[h, arm_h], float(r_all_h[h]),
                        arm_h, h, float(sigma_mat[h, arm_h]))
    return reg_acc


def run_condition(algo_cls, kwargs, pi0, n_seeds, n_jobs):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(run_one_seed)(algo_cls, kwargs, pi0, seed=s)
        for s in range(n_seeds)
    ))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _boxplot_panel(ax, data_dict, xtick_labels, xlabel):
    """
    data_dict    : {algo_name: [array(n_seeds) per tick]}
    xtick_labels : list of str
    """
    n_ticks  = len(xtick_labels)
    width    = 0.28
    offsets  = [-width / 2, width / 2]
    ticks    = np.arange(1, n_ticks + 1)

    for i, name in enumerate(ALGO_NAMES):
        color = ALGO_COLORS[name]
        pos   = ticks + offsets[i]
        ax.boxplot(
            data_dict[name],
            positions    = pos,
            widths       = width,
            patch_artist = True,
            notch        = False,
            medianprops  = dict(color="white", linewidth=2.0),
            whiskerprops = dict(color=color, linewidth=1.2),
            capprops     = dict(color=color, linewidth=1.2),
            flierprops   = dict(marker="o", markersize=3,
                                markerfacecolor=color, alpha=0.4,
                                linestyle="none"),
            boxprops     = dict(facecolor=color, alpha=0.70, linewidth=0),
        )

    ax.set_xticks(ticks)
    ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Final cumulative regret", fontsize=11)

    patches = [mpatches.Patch(facecolor=ALGO_COLORS[n], alpha=0.75, label=n)
               for n in ALGO_NAMES]
    ax.legend(handles=patches, fontsize=9, frameon=True,
              loc="upper left", handlelength=1.2)


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ablation 1 — baseline policy quantile
# ---------------------------------------------------------------------------

def run_ablations(n_seeds, n_jobs):
    ref_env = Environment(**ENV_CFG, drift_fn=DRIFT_FN, seed=42)
    sigma   = 1.92
    lam     = sigma ** 2 / S ** 2

    # --- Ablation 1: quantile ---
    print("\n=== Ablation 1: baseline policy quantile ===")
    data_q = {n: [] for n in ALGO_NAMES}
    for q in QUANTILE_GRID:
        pi0 = ref_env.compute_quantile_pi0(quantile=q)
        print(f"  q={q}")
        for name, cls in [("Dri-MED", DRI_MED), ("Dri-IMED", DRI_IMED)]:
            kw      = dict(d=d, lam=lam, delta=DELTA, S=S, nu0=0.0,
                           epsilon=DEFAULT_EPSILON, gamma_decay=GAMMA_DECAY)
            regrets = run_condition(cls, kw, pi0, n_seeds, n_jobs)
            data_q[name].append(regrets)
            print(f"    {name}  median={np.median(regrets):.1f}")

    # --- Ablation 2: epsilon ---
    print("\n=== Ablation 2: satisficing tolerance epsilon ===")
    pi0      = ref_env.compute_quantile_pi0(quantile=DEFAULT_QUANTILE)
    data_eps = {n: [] for n in ALGO_NAMES}
    for eps in EPSILON_GRID:
        print(f"  epsilon={eps}")
        for name, cls in [("Dri-MED", DRI_MED), ("Dri-IMED", DRI_IMED)]:
            kw      = dict(d=d, lam=lam, delta=DELTA, S=S, nu0=0.0,
                           epsilon=eps, gamma_decay=GAMMA_DECAY)
            regrets = run_condition(cls, kw, pi0, n_seeds, n_jobs)
            data_eps[name].append(regrets)
            print(f"    {name}  median={np.median(regrets):.1f}")

    # --- Single figure with two subplots ---------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    _boxplot_panel(ax1, data_q,
                   xtick_labels=[str(q) for q in QUANTILE_GRID],
                   xlabel=r"Baseline quantile $q$")
    ax1.set_title("Baseline policy", fontsize=10)

    _boxplot_panel(ax2, data_eps,
                   xtick_labels=[str(e) for e in EPSILON_GRID],
                   xlabel=r"Satisficing tolerance $\varepsilon$")
    ax2.set_title("Constraint tightness", fontsize=10)
    ax2.set_ylabel("")   # avoid redundant y-label on right panel

    # remove per-panel legends — replaced by a single shared one
    ax1.get_legend().remove()
    ax2.get_legend().remove()

    patches = [mpatches.Patch(facecolor=ALGO_COLORS[n], alpha=0.75, label=n)
               for n in ALGO_NAMES]
    fig.legend(handles=patches, fontsize=10, frameon=True,
               loc="lower center", ncol=len(ALGO_NAMES),
               bbox_to_anchor=(0.5, -0.18))

    fig.suptitle(
        f"Ablation study  —  Abrupt drift  —  "
        f"$K={ENV_CFG['num_actions']},\\ H={ENV_CFG['num_users']},"
        f"\\ L={ENV_CFG['num_episodes']},"
        f"\\ \\gamma={GAMMA_DECAY}$",
        fontsize=11, fontweight="bold", y=0.925,
    )

    fig.tight_layout()
    save_fig(fig, "ablation")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",  type=int, default=64)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()
    run_ablations(args.seeds, args.n_jobs)