"""
Experiment runner for Dri-MED / Dri-IMED vs stationary baselines.

Figures are saved in figures/ (created automatically).

Three panels per drift regime  →  figures/results_{drift}.{png,pdf}
  Panel 1 — cumulative regret
  Panel 2 — per-episode ESTIMATED violation rate
             (1/H) * sum_h 1{ <theta_hat, phi_{A_{h,l}}> < tau }
  Panel 3 — per-episode TRUE violation rate
             (1/H) * sum_h 1{ <theta*,    phi_{A_{h,l}}> < tau }
"""

import argparse
import inspect
import os

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from environment import Environment, DriftFunctions
from oful        import OFUL
from lin_imed    import LinIMED
from lin_med     import LinMED
from dri_imed    import DRI_IMED
from dri_med     import DRI_MED
from lin_ts import LinTS


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Oracle constraint threshold
# ---------------------------------------------------------------------------

def oracle_tau(env: Environment, pi0: np.ndarray, epsilon: float) -> float:
    """
    tau = (1 - epsilon) * (1/H) * sum_h sum_a pi0[h,a] * phi_{a,h}^T theta*

    Context-free and episode-invariant — a conservative universal threshold
    used to evaluate ALL algorithms on the same footing.
    """
    features    = env.compute_features(env._pref_mean)   # (H, A, d)
    mean_reward = features @ env.theta                    # (H, A)
    baseline    = float((pi0 * mean_reward).sum(axis=1).mean())
    return (1.0 - epsilon) * baseline


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------

def run_one_seed(algo_cls, algo_kwargs, env_cfg, drift_fn,
                 pi0, epsilon, tau_oracle, seed):
    """
    Returns
    -------
    cumreg        (L,)  cumulative regret
    viol_est_rate (L,)  per-episode ESTIMATED violation rate in [0, 1]
                        (1/H) * sum_h 1{ <theta_hat, phi_{A_{h,l}}> < tau }
    viol_true_rate(L,)  per-episode TRUE violation rate in [0, 1]
                        (1/H) * sum_h 1{ <theta*,    phi_{A_{h,l}}> < tau }
    """
    env      = Environment(**env_cfg, drift_fn=drift_fn, seed=42)
    env._rng = np.random.default_rng(seed)

    H = env.num_users
    L = env.num_episodes

    kw = dict(algo_kwargs)
    if "pi0" in inspect.signature(algo_cls.__init__).parameters:
        kw["pi0"] = pi0
    algo = algo_cls(**kw)

    opt_arms = env.optimal_policy()
    rng      = np.random.default_rng(seed)

    cumreg         = np.zeros(L)
    viol_est_rate  = np.zeros(L)
    viol_true_rate = np.zeros(L)
    reg_acc        = 0.0

    for ell in range(L):
        prefs      = env.get_preferences()
        ctx        = env.sample_contexts(ell)
        features   = env.compute_features(prefs)            # (H, A, d)
        sigma_mat  = env.compute_sigma_matrix(ctx, prefs)   # (H, A)
        true_means = features @ env.theta                   # (H, A)

        algo.start_episode(features, sigma_mat)

        viol_est_ep  = 0   # violations within this episode
        viol_true_ep = 0

        dummy_arms = np.zeros(H, dtype=int)
        for h in range(H):
            arm_h         = algo.select_arm(features[h], sigma_mat[h], rng)
            dummy_arms[h] = arm_h

            # --- regret --------------------------------------------------
            opt_a    = int(opt_arms[h])
            reg_acc += max(0.0,
                           float(true_means[h, opt_a]) -
                           float(true_means[h, arm_h]))

            # --- estimated violation  1{ theta_hat^T phi < tau } ---------
            mean_est = float(features[h, arm_h] @ algo.theta_hat)
            if mean_est < tau_oracle:
                viol_est_ep += 1

            # --- true violation  1{ theta*^T phi < tau } -----------------
            if float(true_means[h, arm_h]) < tau_oracle:
                viol_true_ep += 1

            r_all_h, _ = env.compute_rewards(dummy_arms, prefs, ctx)
            algo.update(features[h, arm_h], float(r_all_h[h]),
                        arm_h, h, float(sigma_mat[h, arm_h]))

        cumreg[ell]         = reg_acc
        viol_est_rate[ell]  = viol_est_ep  / H   # fraction of users violated
        viol_true_rate[ell] = viol_true_ep / H

    return cumreg, viol_est_rate, viol_true_rate


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

def run_experiment(algo_cls, algo_kwargs, env_cfg, drift_fn,
                   pi0, epsilon, tau_oracle, n_seeds=10, n_jobs=-1):
    out = Parallel(n_jobs=n_jobs)(
        delayed(run_one_seed)(
            algo_cls, algo_kwargs, env_cfg, drift_fn,
            pi0, epsilon, tau_oracle, seed=s)
        for s in range(n_seeds)
    )
    reg_arr        = np.array([o[0] for o in out])   # (S, L)
    viol_est_arr   = np.array([o[1] for o in out])   # (S, L)
    viol_true_arr  = np.array([o[2] for o in out])   # (S, L)

    for s, (r, ve, vt) in enumerate(zip(reg_arr, viol_est_arr, viol_true_arr)):
        print(f"    seed {s+1}/{n_seeds}  "
              f"regret={r[-1]:.2f}  "
              f"mean_viol_est={ve.mean():.3f}  "
              f"mean_viol_true={vt.mean():.3f}")

    return reg_arr, viol_est_arr, viol_true_arr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS  = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
           '#a65628', '#984ea3', '#999999', '#e41a1c']
MARKERS = ['o', 'v', 's', 'd', '<', '>']

ALGO_STYLE = {
    "OFUL":     dict(linestyle="-",  marker="o"),
    "LinMED":   dict(linestyle="-",  marker="v"),
    "LinIMED":  dict(linestyle="-",  marker="s"),
    "Dri-MED":  dict(linestyle="--", marker="d"),
    "Dri-IMED": dict(linestyle="--", marker="<"),
}


def _plot_band(ax, eps, arr, color, label, style):
    """Median ± 5–95 percentile band."""
    median = np.percentile(arr, 50, axis=0)
    q05    = np.percentile(arr,  5, axis=0)
    q95    = np.percentile(arr, 95, axis=0)
    ax.plot(eps, median, color=color, label=label, linewidth=2.0,
            markevery=0.08, **style)
    ax.fill_between(eps, q05, q95, color=color, alpha=0.12)


def plot_results(results_dict, env_cfg, drift_name,
                 epsilon, tau_oracle, out_prefix):
    """
    Three-panel figure with a single suptitle and a shared horizontal legend.
      Panel 1 — cumulative regret
      Panel 2 — per-episode estimated violation rate
      Panel 3 — per-episode true violation rate
    """
    L   = env_cfg["num_episodes"]
    K   = env_cfg["num_actions"]
    H   = env_cfg["num_users"]
    d   = env_cfg["feature_dim"]
    eps = np.arange(1, L + 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    ax_reg, ax_vest, ax_vtrue = axes

    # draw curves and collect one handle per algo for the shared legend
    handles, labels = [], []
    for i, (name, (reg_arr, vest_arr, vtrue_arr)) in enumerate(results_dict.items()):
        c = COLORS[i % len(COLORS)]
        s = ALGO_STYLE.get(name,
              dict(linestyle="-", marker=MARKERS[i % len(MARKERS)]))
        _plot_band(ax_reg,   eps, reg_arr,   c, name, s)
        _plot_band(ax_vest,  eps, vest_arr,  c, name, s)
        _plot_band(ax_vtrue, eps, vtrue_arr, c, name, s)
        # invisible proxy artist for the shared legend
        h, = ax_reg.plot([], [], color=c, linewidth=2.0, **s)
        handles.append(h)
        labels.append(name)

    # --- common suptitle (environment + drift shown once) ----------------
    fig.suptitle(
        f"{drift_name.capitalize()} Drift  —  "
        f"$K={K},\\ H={H},\\ L={L},\\ d={d}$",
        fontsize=12, fontweight="bold", y=0.9125,
    )

    # --- panel titles (concise) ------------------------------------------
    ax_reg.set_title("Cumulative Regret", fontsize=11)
    ax_reg.set_xlabel("Episode $\\ell$", fontsize=11)
    ax_reg.set_ylabel("Cumulative Regret", fontsize=12)

    ax_vest.set_title(
        "Estimated Violation Rate",
        fontsize=10)
    ax_vest.set_xlabel("Episode $\\ell$", fontsize=11)
    ax_vest.set_ylabel("Violation rate", fontsize=11)
    ax_vest.set_ylim(-0.02, 1.05)
    ax_vest.axhline(0, color="black", linewidth=0.6, linestyle=":")

    ax_vtrue.set_title(
        "True Violation Rate",
        fontsize=10)
    ax_vtrue.set_xlabel("Episode $\\ell$", fontsize=11)
    ax_vtrue.set_ylabel("Violation rate", fontsize=11)
    ax_vtrue.set_ylim(-0.02, 1.05)
    ax_vtrue.axhline(0, color="black", linewidth=0.6, linestyle=":")

    # --- single horizontal legend centred below all panels ---------------
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(results_dict),
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, -0.10),
    )

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = f"{out_prefix}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  saved -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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

EPSILON = 0.1

d     = ENV_CFG["feature_dim"]
S     = ENV_CFG["theta_norm"]
delta = 0.01

DRIFT_FNS = {
    "none":          DriftFunctions.none(),
    "gradual":       DriftFunctions.gradual(100),
    "periodic":      DriftFunctions.periodic(100),
    "abrupt":        DriftFunctions.abrupt(100),
    "decaying":      DriftFunctions.decaying(100),
    "burst":         DriftFunctions.burst(100),
    "regime_shift":  DriftFunctions.regime_shift(100),
    "sawtooth":      DriftFunctions.sawtooth(100),
    "adversarial":   DriftFunctions.adversarial(100),
}

def make_algo_configs(sigma, pi0):
    lam = (sigma ** 2) / (S ** 2)
    return {
        "OFUL":     (OFUL,     dict(d=d, lam=lam, delta=delta, S=S,
                                    sigma_upper=sigma, baseline_arm=0)),
        "LinTS": (LinTS, dict(d=d, lam=lam, delta=delta, S=S, sigma_upper=sigma, baseline_arm=0)),
        "LinMED":   (LinMED,   dict(d=d, lam=lam, delta=delta, S=S,
                                    baseline_arm=0)),
        "LinIMED":  (LinIMED,  dict(d=d, lam=lam, delta=delta, S=S,
                                    C=30, baseline_arm=0)),
        "Dri-MED":  (DRI_MED,  dict(d=d, lam=lam, delta=delta, S=S,
                                    nu0=0.0, epsilon=EPSILON,
                                    gamma_decay=0.99, pi0=pi0)),
        "Dri-IMED": (DRI_IMED, dict(d=d, lam=lam, delta=delta, S=S,
                                    nu0=0.0, epsilon=EPSILON,
                                    gamma_decay=0.99, pi0=pi0)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",  type=int, default=128)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--drifts", nargs="+",
                        default=["none", "gradual", "periodic", "abrupt",]) #"decaying", "burst", "regime_shift", "sawtooth", "adversarial"])
    args = parser.parse_args()

    for drift_name in args.drifts:
        if drift_name not in DRIFT_FNS:
            print(f"Unknown drift '{drift_name}', skipping."); continue

        print(f"\n{'='*55}\nDrift: {drift_name}\n{'='*55}")
        drift_fn = DRIFT_FNS[drift_name]

        ref_env = Environment(**ENV_CFG, drift_fn=drift_fn, seed=42)
        sigma   = 1.92
        pi0     = ref_env.compute_quantile_pi0(quantile=0.50)
        tau     = oracle_tau(ref_env, pi0, EPSILON)
        print(f"  sigma_upper={sigma:.4f}  tau_oracle={tau:.4f}")

        algo_configs = make_algo_configs(sigma, pi0)
        results_all  = {}

        for name, (cls, kwargs) in algo_configs.items():
            print(f"\n  -- {name} --")
            reg_arr, vest_arr, vtrue_arr = run_experiment(
                cls, kwargs, ENV_CFG, drift_fn,
                pi0, EPSILON, tau,
                n_seeds=args.seeds,
                n_jobs=args.n_jobs,
            )
            results_all[name] = (reg_arr, vest_arr, vtrue_arr)

        out_prefix = os.path.join(FIGURES_DIR, f"results_{drift_name}")
        plot_results(
            results_all, ENV_CFG, drift_name,
            epsilon=EPSILON, tau_oracle=tau,
            out_prefix=out_prefix,
        )