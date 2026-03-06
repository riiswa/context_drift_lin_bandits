from typing import Callable
import numpy as np


class Environment:
    """
    Episodic contextual linear bandit with preference feedback and context drift.

    Signal mean  : E[Y_{h,l} | a] = Phi(a) theta*         (context-independent)
    Signal cov   : Cov[Y | a, C_{h,l}] = Sigma_a * (1 + norm(C_{h,l}))
    Scalar reward: r_{h,l} = omega_h^T Y_{h,l}

    Preference modes:
        stochastic_preferences=False (default):
            p_h is fixed for all episodes. Sampled once at init from Dirichlet.
            phi_{a,h} = Phi(a)^T p_h is fixed and known to the learner.

        stochastic_preferences=True:
            omega_h is resampled each episode from Dirichlet.
            The learner observes omega_h before acting.
            p_h = E[omega_h] is still used for the oracle.
    """

    def __init__(
        self,
        feature_dim           : int,
        theta_norm            : float,
        num_actions           : int,
        num_users             : int,
        num_episodes          : int,
        feedback_dim          : int,
        preference_scale      : float,
        context_dim           : int,
        context_noise         : float,
        reward_noise          : float,
        drift_fn              : Callable[[np.ndarray, int], np.ndarray],
        stochastic_preferences: bool = False,
        seed                  : int  = 42,
    ):
        self.num_users              = num_users
        self.num_actions            = num_actions
        self.num_episodes           = num_episodes
        self.feedback_dim           = feedback_dim
        self.context_dim            = context_dim
        self.context_noise          = context_noise
        self.reward_noise           = reward_noise
        self.stochastic_preferences = stochastic_preferences

        self._rng = np.random.default_rng(seed)

        # true parameter theta*, shape (d,)
        theta      = self._rng.standard_normal(feature_dim)
        self.theta = theta / np.linalg.norm(theta) * theta_norm

        # arm feature matrices Phi(a), shape (A, M, d), row-normalised
        Phi      = self._rng.standard_normal((num_actions, feedback_dim, feature_dim))
        norms    = np.linalg.norm(Phi, axis=2, keepdims=True)
        self.Phi = Phi / np.where(norms < 1e-8, 1.0, norms)

        # base covariance per arm, shape (A, M, M), PSD
        # context-dependent version: Sigma_a * (1 + norm(C_{h,l}))
        raw        = self._rng.standard_normal((num_actions, feedback_dim, feedback_dim))
        self.Sigma = np.array([S @ S.T / feedback_dim for S in raw])

        # preference vectors
        # _pref_mean  : p_h = E[omega_h], shape (H, M), used by oracle
        # _pref_alpha : Dirichlet concentration, shape (H, M)
        P                = self._rng.dirichlet(np.ones(feedback_dim), size=num_users)
        self._pref_mean  = P
        self._pref_alpha = P * preference_scale

        # fixed preferences sampled once (used when stochastic_preferences=False)
        self._fixed_preferences = np.array([
            self._rng.dirichlet(self._pref_alpha[h])
            for h in range(num_users)
        ])  # (H, M)

        # context drift: mu_l = drift_fn(l) * direction, shape (L, context_dim)
        # direction          = self._rng.standard_normal(context_dim)
        # direction          = direction / np.linalg.norm(direction)
        # ell                = np.arange(num_episodes)
        # self.context_means = drift_fn(ell, num_episodes)[:, None] * direction
        ell = np.arange(num_episodes)
        self.context_scales = 1.0 + drift_fn(ell, num_episodes)


    def get_preferences(self) -> np.ndarray:
        if self.stochastic_preferences:
            return np.array([
                self._rng.dirichlet(self._pref_alpha[h])
                for h in range(self.num_users)
            ])  # (H, M)
        return self._fixed_preferences  # same object every call

    def compute_features(self, preferences: np.ndarray) -> np.ndarray:
        # preferences @ Phi[a] : (H, M) @ (M, d) -> (H, d)
        return np.stack(
            [preferences @ self.Phi[a] for a in range(self.num_actions)],
            axis=1,
        )  # (H, A, d)

    # def sample_contexts(self, episode: int) -> np.ndarray:
    #     return self._rng.normal(
    #         self.context_means[episode], self.context_noise,
    #         (self.num_users, self.context_dim),
    #     )
    def sample_contexts(self, episode: int) -> np.ndarray:
        return self._rng.normal(
            0,
            self.context_noise * self.context_scales[episode],
            (self.num_users, self.context_dim),
        )

    def compute_rewards(
        self,
        actions     : np.ndarray,   # (H,)
        preferences : np.ndarray,   # (H, M)
        contexts    : np.ndarray,   # (H, context_dim)
    ) -> tuple[np.ndarray, np.ndarray]:
        features   = self.compute_features(preferences)                  # (H, A, d)
        all_means  = features @ self.theta                               # (H, A)
        true_means = all_means[np.arange(self.num_users), actions]      # (H,)

        context_norms = np.linalg.norm(contexts, axis=1)                # (H,)

        rewards = np.empty(self.num_users)
        for h in range(self.num_users):
            a = actions[h]
            # context-scaled covariance: Sigma_a * reward_noise * (1 + norm(C_{h,l}))
            Sig_ahc = self.Sigma[a] * self.reward_noise * (1.0 + context_norms[h])
            # draw M-dimensional signal vector with context-independent mean
            Y_h = self._rng.multivariate_normal(
                mean=self.Phi[a] @ self.theta,
                cov=Sig_ahc,
            )
            rewards[h] = preferences[h] @ Y_h

        return rewards, true_means

    def compute_episode_noise_bound(
        self,
        contexts    : np.ndarray,   # (H, context_dim)
        preferences : np.ndarray,   # (H, M)
    ) -> float:
        """Returns max_{h,a} omega_h^T Sigma_{a,h,l} omega_h (Eq. 6)."""
        context_norms = np.linalg.norm(contexts, axis=1)   # (H,)
        sigma_sq = np.array([
            [
                preferences[h] @
                (self.Sigma[a] * self.reward_noise * (1.0 + context_norms[h])) @
                preferences[h]
                for a in range(self.num_actions)
            ]
            for h in range(self.num_users)
        ])  # (H, A)
        return float(sigma_sq.max())

    def compute_sigma_matrix(
        self,
        contexts    : np.ndarray,   # (H, context_dim)
        preferences : np.ndarray,   # (H, M)
    ) -> np.ndarray:
        """Returns sigma_{a,h,l} = sqrt(p_h^T Sigma_{a,C_{h,l}} p_h) as shape (H, A)."""
        context_norms = np.linalg.norm(contexts, axis=1)   # (H,)
        sigma_sq = np.array([
            [
                preferences[h] @
                (self.Sigma[a] * self.reward_noise * (1.0 + context_norms[h])) @
                preferences[h]
                for a in range(self.num_actions)
            ]
            for h in range(self.num_users)
        ])  # (H, A)
        return np.sqrt(np.maximum(sigma_sq, 1e-12))


    # def compute_sigma_upper(self, confidence: float = 2.0) -> float:
    #     """
    #     Analytical upper bound on sigma_{a,h,l} for all arms, users, and episodes.
    #
    #     sigma^2_{a,h,l} = p_h^T * Sigma_a * reward_noise * (1 + ||C_{h,l}||) * p_h
    #
    #     Bounded by three independent factors:
    #
    #     1. max_a max_i Sigma_a[i,i]:
    #        p^T Sigma_a p is convex in p, so its max over the probability simplex
    #        is attained at a vertex e_i, giving max_i Sigma_a[i,i].
    #
    #     2. max (1 + ||C_{h,l}||):
    #        C_{h,l} ~ N(context_means[l], context_noise^2 * I_D).
    #        Chi distribution tail: ||C - mu|| <= context_noise*(sqrt(D) + confidence)
    #        with probability >= 1 - exp(-confidence^2/2).
    #        So: ||C_{h,l}|| <= max_l ||context_means[l]|| + context_noise*(sqrt(D)+confidence)
    #
    #     3. reward_noise scalar.
    #     """
    #     max_drift        = float(np.max(np.linalg.norm(self.context_means, axis=1)))
    #     max_context_norm = max_drift + self.context_noise * (np.sqrt(self.context_dim) + confidence)
    #     max_quad         = max(float(self.Sigma[a].diagonal().max()) for a in range(self.num_actions))
    #     return float(np.sqrt(self.reward_noise * (1.0 + max_context_norm) * max_quad))

    def compute_sigma_upper(self, confidence: float = 2.0) -> float:
        # C_{h,l} ~ N(0, (context_noise * scale_l)^2 * I_D)
        # ||C_{h,l}|| <= context_noise * max_scale * (sqrt(D) + confidence) w.h.p.
        max_scale = float(np.max(self.context_scales))
        max_context_norm = self.context_noise * max_scale * (np.sqrt(self.context_dim) + confidence)
        max_quad = max(float(self.Sigma[a].diagonal().max()) for a in range(self.num_actions))
        return float(np.sqrt(self.reward_noise * (1.0 + max_context_norm) * max_quad))

    def optimal_policy(self) -> np.ndarray:
        """Oracle best arm per user using mean preference p_h. Returns shape (H,)."""
        features = self.compute_features(self._pref_mean)   # (H, A, d)
        return (features @ self.theta).argmax(axis=1)       # (H,)

    def compute_quantile_pi0(self, quantile=0.5):
        """
        Returns pi0 of shape (H, A): one-hot per user at the given performance quantile.
        Each user's arm is chosen based on their own preference vector.
        """
        features = self.compute_features(self._pref_mean)  # (H, A, d)
        mean_rewards = features @ self.theta  # (H, A)

        H, A = mean_rewards.shape
        pi0 = np.zeros((H, A))

        for h in range(H):
            threshold = np.quantile(mean_rewards[h], quantile)
            baseline_arm = int(np.argmin(np.abs(mean_rewards[h] - threshold)))
            pi0[h, baseline_arm] = 1.0

        return pi0

# ---------------------------------------------------------------------------
# Drift functions
# ---------------------------------------------------------------------------

class DriftFunctions:
    """
    Built-in drift trajectories. Each returns a callable f(ell, num_episodes)
    that produces a shape (L,) array of drift magnitudes.

    Usage:
        env = Environment(..., drift_fn=DriftFunctions.gradual(magnitude=2.0))
    """

    @staticmethod
    def none():
        def f(ell, num_episodes):
            return np.zeros(len(ell))
        return f

    @staticmethod
    def gradual(magnitude: float = 1.0):
        def f(ell, num_episodes):
            return magnitude * (ell / num_episodes)
        return f

    @staticmethod
    def periodic(magnitude: float = 1.0):
        def f(ell, num_episodes):
            return magnitude * (1 + np.sin(2 * np.pi * ell / (num_episodes // 50))) / 2
        return f

    @staticmethod
    def abrupt(magnitude: float = 1.0, change_points: list[int] | None = None):
        def f(ell, num_episodes):
            cps = change_points or [num_episodes // 3, 2 * num_episodes // 3]
            levels = np.zeros(len(ell))
            for cp in cps:
                levels[ell >= cp] = magnitude  # flat, not growing
            return levels

        return f

    @staticmethod
    def decaying(magnitude: float = 1.0, rate: float = 5.0):
        """
        Exponential decay: high noise early, low noise late.
        Stresses the exploration phase when theta_hat is least certain.
        Best for revealing differences in early exploration strategy.
        """

        def f(ell, num_episodes):
            return magnitude * np.exp(-rate * ell / num_episodes)

        return f

    @staticmethod
    def burst(magnitude: float = 1.0, centers: list[float] | None = None,
              width: float = 0.05):
        """
        Gaussian noise bursts at specified fractional positions.
        Default: three bursts at 20%, 50%, 80% of episodes.
        Each burst temporarily overwhelms estimation then subsides.
        Tests how quickly algorithms recover after noise spikes.
        """

        def f(ell, num_episodes):
            cs = centers or [0.2, 0.5, 0.8]
            out = np.zeros(len(ell))
            t = ell / num_episodes
            for c in cs:
                out += magnitude * np.exp(-((t - c) ** 2) / (2 * width ** 2))
            return out

        return f

    @staticmethod
    def regime_shift(magnitude: float = 1.0, num_regimes: int = 5):
        """
        Random-walk regime shifts: noise level jumps to a new random level
        every L/num_regimes episodes, with the new level drawn proportionally
        to magnitude. Unlike abrupt (always increasing), regimes can go up or
        down — testing whether algorithms adapt in both directions.
        """

        def f(ell, num_episodes):
            rng = np.random.default_rng(0)  # fixed seed for reproducibility
            boundaries = np.linspace(0, num_episodes, num_regimes + 1, dtype=int)
            levels = rng.uniform(0, magnitude, size=num_regimes)
            out = np.zeros(len(ell))
            for i in range(num_regimes):
                mask = (ell >= boundaries[i]) & (ell < boundaries[i + 1])
                out[mask] = levels[i]
            return out

        return f

    @staticmethod
    def sawtooth(magnitude: float = 1.0, num_cycles: int = 5):
        """
        Linearly ramps up then resets sharply, repeating num_cycles times.
        Combines gradual increase with abrupt resets — tests whether algorithms
        exploit low-noise periods and survive sudden noise jumps back to baseline.
        """

        def f(ell, num_episodes):
            t = ell / num_episodes
            return magnitude * (t * num_cycles % 1.0)

        return f

    @staticmethod
    def adversarial(magnitude: float = 1.0):
        """
        Noise spikes whenever the algorithm is likely to be most confident:
        low early (let theta_hat build), then bursts at regular intervals
        timed to disrupt exploitation phases.
        High noise at episodes 100, 300, 600, 900 — just after typical
        convergence points — then decays back quickly.
        """

        def f(ell, num_episodes):
            spike_fracs = [0.1, 0.3, 0.6, 0.9]
            width = 0.03
            t = ell / num_episodes
            out = np.zeros(len(ell))
            for c in spike_fracs:
                out += magnitude * np.exp(-((t - c) ** 2) / (2 * width ** 2))
            return out

        return f


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    L, H, A, d, M, C = 50, 10, 6, 8, 3, 5

    for stoch in [False, True]:
        env = Environment(
            feature_dim           = d,
            theta_norm            = 1.0,
            num_actions           = A,
            num_users             = H,
            num_episodes          = L,
            feedback_dim          = M,
            preference_scale      = 2.0,
            context_dim           = C,
            context_noise         = 0.1,
            reward_noise          = 0.5,
            drift_fn              = DriftFunctions.gradual(magnitude=1.0),
            stochastic_preferences= stoch,
            seed                  = 0,
        )

        pref0 = env.get_preferences()
        pref1 = env.get_preferences()
        prefs_same = np.allclose(pref0, pref1)

        print(f"\nstochastic_preferences={stoch}")
        print(f"  Preferences same across calls : {prefs_same}  (expected {not stoch})")

        for episode in range(3):
            prefs   = env.get_preferences()
            ctx     = env.sample_contexts(episode)
            actions = np.zeros(H, dtype=int)
            rews, means = env.compute_rewards(actions, prefs, ctx)
            sigma_hat   = env.compute_episode_noise_bound(ctx, prefs)

            print(f"  ep={episode}  rewards={np.round(rews,2)}  "
                  f"means={np.round(means,2)}  sigma_hat={sigma_hat:.3f}")

        opt = env.optimal_policy()
        print(f"  Oracle arms: {opt}")