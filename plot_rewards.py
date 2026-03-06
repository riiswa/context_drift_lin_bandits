import numpy as np
import matplotlib.pyplot as plt

from environment import Environment, DriftFunctions

env = Environment(
    feature_dim      = 4,
    theta_norm       = 1.0,
    num_actions      = 5,
    num_users        = 10,
    num_episodes     = 100,
    feedback_dim     = 4,
    preference_scale = 2.0,
    context_dim      = 2,
    context_noise    = 1.0,
    reward_noise     = 0.1,
    drift_fn         = DriftFunctions.sawtooth(magnitude=100),
    seed             = 42,
)

N_SAMPLES = 30   # reward draws per episode per action

# --- identify best and worst actions via oracle ---------------------------
features  = env.compute_features(env._pref_mean)
mean_rews = features @ env.theta
best_arm  = int(np.argmax(mean_rews.mean(axis=0)))
worst_arm = int(np.argmin(mean_rews.mean(axis=0)))
print(f"Best arm: {best_arm}  Worst arm: {worst_arm}")

actions_to_plot = [best_arm, worst_arm]

data = {a: {"rew_x": [], "rew_y": [], "true_mean": [], "sigma": []}
        for a in actions_to_plot}

for ell in range(env.num_episodes):
    prefs     = env.get_preferences()
    ctx       = env.sample_contexts(ell)
    sigma_mat = env.compute_sigma_matrix(ctx, prefs)

    for a in actions_to_plot:
        arms          = np.full(env.num_users, a, dtype=int)
        _, true_means = env.compute_rewards(arms, prefs, ctx)

        data[a]["true_mean"].append(true_means.mean())
        data[a]["sigma"].append(sigma_mat[:, a].mean())

        # multiple draws to densify scatter
        for _ in range(N_SAMPLES):
            rews, _ = env.compute_rewards(arms, prefs, ctx)
            data[a]["rew_x"].append(ell)
            data[a]["rew_y"].append(float(rews.mean()))

# --- baseline sigma (zero context, no drift) ------------------------------
base_ctx   = np.zeros((env.num_users, env.context_dim))
base_sigma = env.compute_sigma_matrix(base_ctx, env.get_preferences())
baseline   = {a: float(base_sigma[:, a].mean()) for a in actions_to_plot}

# --- plot -----------------------------------------------------------------
labels = {best_arm: "Best arm", worst_arm: "Worst arm"}
colors = {best_arm: "#377eb8", worst_arm: "#e41a1c"}
eps    = np.arange(env.num_episodes)

fig, ax = plt.subplots(figsize=(12, 4))

for a in actions_to_plot:
    c         = colors[a]
    lbl       = labels[a]
    true_mean = np.array(data[a]["true_mean"])
    s0        = baseline[a]
    mu        = true_mean.mean()

    ax.scatter(data[a]["rew_x"], data[a]["rew_y"],
               color=c, alpha=0.4, s=3, linewidths=0)
    ax.plot(eps, true_mean, color=c, linewidth=2.0, label=f"{lbl} (true mean)")
    ax.axhline(mu + s0, color=c, linewidth=1.2, linestyle="--", label=f"{lbl} ±σ₀")
    ax.axhline(mu - s0, color=c, linewidth=1.2, linestyle="--")

ax.set_ylabel("Reward")
ax.set_xlabel("Episode")
ax.set_title("Reward evolution — true mean ± baseline σ (no drift inflation)")
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig("reward_evolution.png", dpi=150)
print("saved -> reward_evolution.png")
plt.show()