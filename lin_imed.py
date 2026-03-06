import numpy as np
from base_agent import BaseLinBandit
from utils import randargmin


class LinIMED(BaseLinBandit):
    """
    Lin-IMED version 3, faithfully matching the reference Lin_IMED implementation.

    Key fix vs previous version
    ---------------------------
    The reference always uses R (= noise std) as a FIXED constant in beta_t:

        beta_t = (R * sqrt(logdetV + 2*log(1/delta)) + sqrt(lam)*S)^2

    Since lam = R^2/S^2, both terms equal R at initialisation (inner=0),
    but after data accumulates inner > 0 and the first term grows as R*sqrt(inner).

    The previous code left self._sigma = 1.0 (BaseLinBandit default), making
    the first term 1.0*sqrt(inner) instead of R*sqrt(inner) — up to 70x too large
    after sufficient data.

    Fix: recover R = sqrt(lam) * S and call _set_sigma(R) once at __init__.
    sigma is then frozen for the entire run, exactly like the reference.

    Index (version 3)
    -----------------
    Sub-optimal arms:
        idx_a = Delta_a^2 / (beta_t * lev_a) - log(beta_t * lev_a)

    Best-UCB arm special case:
        idx_best = min(log(C / Delta_worst^2), -log(beta_t * lev_best))

    where:
        lev_a   = phi_a^T V_t^{-1} phi_a          (leverage score)
        Delta_a = UCB_best - UCB_a                 (UCB gap, >= 0)
        best    = argmax UCB,   worst = argmin UCB
    """

    def __init__(self, d, lam, delta, S=1.0, C=30, baseline_arm=0):
        super().__init__(d, lam, delta, S, baseline_arm)
        self.C = C

        # Fix: set sigma = R = sqrt(lam)*S, frozen for the entire run.
        # This makes beta_t = (R*sqrt(inner) + R)^2, matching the reference exactly.
        R = float(np.sqrt(lam) * S)
        self._set_sigma(R)   # writes self._sigma = R and recomputes beta_t

    def select_arm(self, features_h, sigma_vec, rng=None):
        """
        sigma_vec is accepted for interface compatibility but intentionally
        ignored — the reference uses a fixed declared noise R, never the
        observed per-arm noise.
        """
        A = features_h.shape[0]

        # FIX: cold-start — match reference which returns a random arm at t==1
        # before any data has been collected.
        if self.t == 1:
            return int(rng.integers(A)) if rng is not None else int(np.random.randint(A))

        # --- leverage scores and UCBs ------------------------------------
        lev = np.array([
            max(float(features_h[a] @ self.invVt @ features_h[a]), 1e-12)
            for a in range(A)
        ])
        mu_hat = features_h @ self.theta_hat          # (A,)
        ucb    = mu_hat + np.sqrt(self.beta_t * lev)  # (A,)

        best  = int(np.argmax(ucb))   # best  UCB arm
        worst = int(np.argmin(ucb))   # worst UCB arm

        # UCB gap: Delta_a = UCB_best - UCB_a  (>= 0 for all a)
        gap = ucb[best] - ucb

        # --- IMED index --------------------------------------------------
        idx = np.empty(A)
        for a in range(A):
            denom  = max(self.beta_t * lev[a], 1e-12)
            idx[a] = gap[a] ** 2 / denom - np.log(denom)

        # Best-UCB arm special case (version 3)
        denom_best = max(self.beta_t * lev[best], 1e-12)
        idx[best]  = min(
            np.log(self.C / max(gap[worst] ** 2, 1e-12)),
            -np.log(denom_best),
        )

        return int(randargmin(idx, rng))