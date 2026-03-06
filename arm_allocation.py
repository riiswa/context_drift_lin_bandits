"""
Arm allocation comparison: LinMED vs Dri-MED vs pi_0.

For each user h in [H], a grouped bar chart shows the fraction of pulls
allocated to each arm over the full run, averaged across seeds.
Layout: 2 x 5 subplots (one per user), saved in figures/.
"""

import inspect
import os

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from environment import Environment, DriftFunctions
from lin_med     import LinMED
from dri_med     import DRI_MED

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# aligned with run_experiments.py: LinMED=index1, Dri-MED=index4
BAR_COLORS = {
    "LinMED":  "#ff7f00",
    "Dri-MED": "#a65628",
    r"$\pi_0$": "#4daf4a",
}

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linestyle":    ":",
    "grid.alpha":        0.5,
    "axes.axisbelow":    True,
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

DELTA         = 0.01
EPSILON       = 0.10
GAMMA_DECAY   = 0.99
DEFAULT_QUANT = 0.50
d             = ENV_CFG["feature_dim"]
S             = ENV_CFG["theta_norm"]

# ---------------------------------------------------------------------------
# Runner — returns arm counts per user, shape (H, A)
# ---------------------------------------------------------------------------

def run_one_seed(algo_cls, algo_kwargs, pi0, seed):
    env      = Environment(**ENV_CFG, drift_fn=DRIFT_FN, seed=42)
    env._rng = np.random.default_rng(seed)

    H = env.num_users
    A = env.num_actions
    L = env.num_episodes

    kw = dict(algo_kwargs)
    if "pi0" in inspect.signature(algo_cls.__init__).parameters:
        kw["pi0"] = pi0
    algo = algo_cls(**kw)

    rng        = np.random.default_rng(seed)
    counts     = np.zeros((H, A), dtype=int)   # (H, A)
    dummy_arms = np.zeros(H, dtype=int)

    for ell in range(L):
        prefs     = env.get_preferences()
        ctx       = env.sample_contexts(ell)
        features  = env.compute_features(prefs)
        sigma_mat = env.compute_sigma_matrix(ctx, prefs)

        algo.start_episode(features, sigma_mat)
        for h in range(H):
            arm_h         = algo.select_arm(features[h], sigma_mat[h], rng)
            dummy_arms[h] = arm_h
            counts[h, arm_h] += 1
            r_all_h, _    = env.compute_rewards(dummy_arms, prefs, ctx)
            algo.update(features[h, arm_h], float(r_all_h[h]),
                        arm_h, h, float(sigma_mat[h, arm_h]))
    return counts   # (H, A)


def run_algo(algo_cls, kwargs, pi0, n_seeds, n_jobs):
    """Returns mean allocation fraction over seeds, shape (H, A)."""
    all_counts = np.array(Parallel(n_jobs=n_jobs)(
        delayed(run_one_seed)(algo_cls, kwargs, pi0, seed=s)
        for s in range(n_seeds)
    ))   # (S, H, A)
    mean_counts = all_counts.mean(axis=0)                   # (H, A)
    return mean_counts / mean_counts.sum(axis=1, keepdims=True)   # (H, A) fractions


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_allocation(alloc_dict, pi0, env_cfg, opt_arms, mean_rewards, out_prefix):
    """
    alloc_dict   : {"LinMED": (H,A), "Dri-MED": (H,A)}
    pi0          : (H, A)  baseline policy
    opt_arms     : (H,)    oracle best arm per user
    mean_rewards : (H, A)  true mean reward per user per arm
    """
    H = env_cfg["num_users"]
    A = env_cfg["num_actions"]

    names   = list(alloc_dict.keys()) + [r"$\pi_0$"]
    x       = np.arange(A)
    n_bars  = len(names)
    width   = 0.22
    offsets = np.linspace(-(n_bars - 1) * width / 2,
                           (n_bars - 1) * width / 2, n_bars)

    fig, axes = plt.subplots(5, 2, figsize=(6, 8), sharey=True)
    axes_flat = axes.flatten()

    for h in range(H):
        ax = axes_flat[h]

        for i, name in enumerate(names):
            if name == r"$\pi_0$":
                vals = pi0[h]
            else:
                vals = alloc_dict[name][h]
            ax.bar(x + offsets[i], vals, width,
                   color=BAR_COLORS[name], alpha=0.75, label=name)

        ax.set_title(f"User {h + 1}", fontsize=9)
        ax.set_xticks(x)
        sorted_arms = sorted(range(A), key=lambda a: -mean_rewards[h, a])
        rank = {sorted_arms[0]: "★"}           # best arm
        if A > 1: rank[sorted_arms[1]] = "⁽¹⁾" # 1st suboptimal
        if A > 2: rank[sorted_arms[2]] = "⁽²⁾" # 2nd suboptimal
        tick_labels = [
            f"$a_{{{a+1}}}${rank.get(a, '')}\n({mean_rewards[h, a]:.2f})"
            for a in range(A)
        ]
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_ylim(0, 1.05)
        if h % 2 == 0:
            ax.set_ylabel("Pull fraction", fontsize=8)

    # shared horizontal legend
    handles = [plt.Rectangle((0, 0), 1, 1,
                              facecolor=BAR_COLORS[n], alpha=0.75)
               for n in names]
    fig.legend(handles, names,
               loc="lower center", ncol=n_bars,
               fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        "Arm allocation per user  —  LinMED vs Dri-MED vs $\\pi_0$",
        fontsize=11, fontweight="bold", y=1.02,
    )

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = f"{out_prefix}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",  type=int, default=32)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    ref_env = Environment(**ENV_CFG, drift_fn=DRIFT_FN, seed=42)
    sigma   = 1.92
    lam     = sigma ** 2 / S ** 2
    pi0     = ref_env.compute_quantile_pi0(quantile=DEFAULT_QUANT)

    base_kw = dict(d=d, lam=lam, delta=DELTA, S=S)

    opt_arms     = ref_env.optimal_policy()
    features     = ref_env.compute_features(ref_env._pref_mean)   # (H, A, d)
    mean_rewards = features @ ref_env.theta                        # (H, A)

    print("Running LinMED ...")
    alloc_linmed = run_algo(
        LinMED,
        dict(**base_kw, baseline_arm=0),
        pi0, args.seeds, args.n_jobs,
    )

    print("Running Dri-MED ...")
    alloc_drimed = run_algo(
        DRI_MED,
        dict(**base_kw, nu0=0.0, epsilon=EPSILON, gamma_decay=GAMMA_DECAY),
        pi0, args.seeds, args.n_jobs,
    )

    plot_allocation(
        {"LinMED": alloc_linmed, "Dri-MED": alloc_drimed},
        pi0, ENV_CFG, opt_arms, mean_rewards,
        out_prefix=os.path.join(FIGURES_DIR, "arm_allocation"),
    )