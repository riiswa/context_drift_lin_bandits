import numpy as np
from base_agent import BaseLinBandit
from utils import randargmin


class DRI_IMED(BaseLinBandit):
    """
    Dri-IMED: Algorithm 1 with IMED arm selection step.

    Episode structure (Alg. 1):
      Line 3  : V_{0,l} = gamma * V_{l-1}, theta_{0,l} = theta_{l-1} / gamma  [episode-start discount]
      Line 5  : phi_{a,h,l} = Phi(a)^T p_h / sigma_{a,h,l}                    [per-arm sigma scaling]
      Lines 8-10: compute tau using V_{l-1} and theta_{l-1} (BEFORE discount)
      Lines 12-17: per-user: select arm, observe r, update xi/nu, update V/theta/N_a
    """

    def __init__(self, d, lam, delta, S=1.0, nu0=1.0, epsilon=0.1,
                 gamma_decay=0.99, pi0=None):
        super().__init__(d, lam, delta, S)
        self.nu           = nu0
        self.epsilon      = epsilon
        self.gamma_decay  = gamma_decay
        self.pi0          = pi0
        self.tau          = 0.0
        self.N_a          = None   # lazily initialized (need A)

    # ------------------------------------------------------------------
    # Beta: (sqrt(log det(V)/det(V_0) + 2*log(1/delta)) + sqrt(lam)*S)^2
    # Used for tau (Alg. 1 line 8) and within-episode selection (IMED step line 1)
    # ------------------------------------------------------------------

    def _compute_beta(self):
        det_ratio = self.logdetV - self.d * np.log(self.lam)
        inner     = max(det_ratio + 2.0 * np.log(1.0 / max(self.delta, 1e-300)), 0.0)
        return (np.sqrt(inner) + np.sqrt(self.lam) * self.S) ** 2

    # ------------------------------------------------------------------
    # Episode start  (Alg. 1 lines 3-10)
    # ------------------------------------------------------------------

    def start_episode(self, features, sigma_mat):
        """
        features  : (H, A, d)  unscaled arm features
        sigma_mat : (H, A)     per-arm per-user noise std sigma_{a,h,l}

        Order:
          1. Scale features per arm and user: phi_{a,h,l} = Phi(a)^T p_h / sigma_{a,h,l}  (line 5)
          2. Compute tau with V_{l-1} pre-discount  (lines 8-10)
          3. Apply episode-start discount: V_{0,l} = gamma * V_{l-1}  (line 3)
             N_a is discounted consistently with V to preserve the IMED index scale.
        """
        H, A, _ = features.shape

        if self.N_a is None:
            self.N_a = np.zeros(A)
        if self.pi0 is None:
            raise ValueError("pi0 must be provided to DRI_IMED (required by the algorithm)")
        elif self.pi0.ndim == 1:
            self.pi0 = np.tile(self.pi0, (H, 1))

        # Alg. 1 line 5: phi_{a,h,l} = Phi(a)^T p_h / sigma_{a,h,l}  — per-arm scaling
        features_s = features / np.maximum(sigma_mat, 1e-12)[:, :, None]  # (H, A, d)

        # Alg. 1 lines 8-10: tau computed with V_{l-1} BEFORE discount
        beta_tau = self._compute_beta()
        phi_bar  = np.einsum("ha,had->d", self.pi0, features_s) / H
        mu0_plus = (float(phi_bar @ self.theta_hat)
                    + np.sqrt(beta_tau * max(float(phi_bar @ self.invVt @ phi_bar), 0.0)))
        self.tau = (1.0 - self.epsilon) * mu0_plus

        # Alg. 1 line 3: V_{0,l} = gamma * V_{l-1}, theta_{0,l} = theta_{l-1} / gamma
        # N_a is discounted alongside V so that the IMED index N_a * gap^2 / 2
        # remains consistent with the geometrically-decayed confidence ellipsoid.
        g             = self.gamma_decay
        self.Vt      *= g
        #self.N_a     *= g
        self.logdetV += self.d * np.log(g)
        try:
            self.invVt = np.linalg.inv(self.Vt)
        except np.linalg.LinAlgError:
            self.invVt = np.linalg.pinv(self.Vt)
        self.theta_hat = self.invVt @ self.XTy  # = theta_{l-1} / gamma

    # ------------------------------------------------------------------
    # IMED arm selection  (IMED step algorithm)
    # Called with the pre-update state (theta_hat, invVt, N_a) for user h,
    # i.e. before this user's observation is incorporated into V.
    # ------------------------------------------------------------------

    def select_arm(self, features_h, sigma_vec, rng=None):
        """
        features_h : (A, d)  unscaled features for user h
        sigma_vec  : (A,)    per-arm noise std sigma_{a,h,l} for this user
        rng        : unused (deterministic); kept for uniform interface

        Called BEFORE update() for user h, so theta_hat and invVt reflect
        V_{h-1,l} (the state after users 1..h-1 have been processed).
        """
        features_s = features_h / np.maximum(sigma_vec, 1e-12)[:, None]  # (A, d)
        A          = features_s.shape[0]

        beta      = self._compute_beta()   # from current V_{h,l}
        sqrt_beta = np.sqrt(beta)

        means  = features_s @ self.theta_hat
        best   = int(np.argmax(means))
        mu_max = float(means[best])

        ucb = np.zeros(A)
        lcb = np.zeros(A)
        for a in range(A):
            lev    = max(float(features_s[a] @ self.invVt @ features_s[a]), 0.0)
            width  = sqrt_beta * np.sqrt(lev)
            ucb[a] = means[a] + width
            lcb[a] = means[a] - width

        # Lagrangian-augmented IMED index (IMED step lines 2-5)
        # sigma_sq = 1: features are pre-scaled by sigma_{a,h,l}, effective noise = 1
        # Arms where UCB(a) < mu_max and a != best are provably sub-optimal given
        # current estimates; their index stays inf so they are never selected.
        idx = np.full(A, np.inf)
        for a in range(A):
            penalty = self.nu * max(0.0, self.tau - lcb[a])
            if a == best:
                idx[a] = penalty
            elif ucb[a] >= mu_max:
                gap    = ucb[a] - mu_max
                idx[a] = self.N_a[a] * gap ** 2 / 2.0 + penalty

        return int(randargmin(idx, rng))

    # ------------------------------------------------------------------
    # Update: nu (lines 14-15) then V/theta/N_a (lines 16-17)
    # ------------------------------------------------------------------

    def update(self, phi, r_tilde, arm, h, sigma_val):
        """
        phi      : unscaled feature for the chosen arm, shape (d,)
        r_tilde  : observed reward minus baseline reward
        arm      : chosen arm index
        h        : user index within episode (0-indexed), used for eta_h step size
        sigma_val: sigma_{arm,h,l} for the chosen arm (scalar)

        Violation xi uses theta_hat BEFORE this user's ridge update (Alg. 1 line 14),
        i.e. the same state that was used in select_arm.
        """
        phi_s = phi / max(sigma_val, 1e-12)
        r_s   = r_tilde / max(sigma_val, 1e-12)

        # Alg. 1 lines 14-15: violation check and nu update with pre-update theta
        mean_before = float(phi_s @ self.theta_hat)
        eta         = 1.0 / np.sqrt(max(h + 1, 1))
        violation   = 1.0 if mean_before < self.tau else 0.0
        self.nu     = max(0.0, self.nu + eta * (violation - self.epsilon / (1.0 - self.epsilon)))

        # Alg. 1 lines 16-17: standard rank-1 update (no per-step gamma)
        self._ridge_update(phi_s, r_s)
        self.N_a[arm] += 1
