import numpy as np


def _beta_sq(sigma, lam, delta, S, logdetV):
    # logdetV is kept as the *incremental* log-det (starting from 0, not d*log(lam)).
    # This makes the inner term equivalent to the reference formula:
    #   logdetV_ref - d*log(lam) + 2*log(1/delta^2)
    # because logdetV_ref = d*log(lam) + incremental, so the d*log(lam) terms cancel.
    inner = max(logdetV + 2.0 * np.log(1.0 / max(delta, 1e-300)), 1e-12)
    return (sigma * np.sqrt(inner) + np.sqrt(lam) * S) ** 2


class BaseLinBandit:
    """Ridge regression state with Sherman-Morrison updates and unified episode interface."""

    def __init__(self, d, lam, delta, S=1.0, baseline_arm=0):
        self.d            = d
        self.lam          = lam
        self.delta        = delta
        self.S            = S
        self.baseline_arm = baseline_arm

        self.theta_hat = np.zeros(d)
        self.Vt        = lam * np.eye(d)
        self.invVt     = np.eye(d) / lam
        self.XTy       = np.zeros(d)
        # FIX: start at 0 (incremental log-det) rather than d*log(lam).
        # The reference formula is  logdetV_ref - d*log(lam) + 2*log(1/delta^2),
        # so initialising here at 0 makes _beta_sq correct without any signature change.
        self.logdetV   = 0
        self.t         = 1
        self._sigma    = 1.0
        self.beta_t    = _beta_sq(1.0, lam, delta, S, self.logdetV)

    def _set_sigma(self, sigma):
        self._sigma = sigma
        self.beta_t = _beta_sq(sigma, self.lam, self.delta, self.S, self.logdetV)

    def _ridge_update(self, phi, r_tilde):
        self.XTy     += r_tilde * phi
        self.Vt      += np.outer(phi, phi)
        tmp           = self.invVt @ phi
        c             = float(phi @ tmp)
        self.logdetV += np.log1p(c)
        self.invVt   -= np.outer(tmp, tmp) / (1.0 + c)
        self.theta_hat = self.invVt @ self.XTy
        self.beta_t   = _beta_sq(self._sigma, self.lam, self.delta, self.S, self.logdetV)
        self.t       += 1

    def start_episode(self, features, sigma_mat):
        """No-op for stationary agents.
        features  : (H, A, d) unscaled arm features
        sigma_mat : (H, A) per-arm per-user noise std sigma_{a,h,l}
        """

    def select_arm(self, features_h, sigma_vec, rng=None):
        """
        features_h : (A, d) unscaled features for user h
        sigma_vec  : (A,)  per-arm noise std sigma_{a,h,l} for this user
        """
        raise NotImplementedError

    def update(self, phi, r_tilde, arm, h, sigma_val):
        """
        phi      : unscaled feature vector of chosen arm, shape (d,)
        r_tilde  : observed reward minus baseline reward
        arm      : index of chosen arm
        h        : user index within the episode (0-indexed)
        sigma_val: noise std sigma_{arm,h,l} for the chosen arm (scalar)
        Stationary agents ignore sigma_val (homoscedastic regression).
        """
        self._ridge_update(phi, r_tilde)