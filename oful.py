import numpy as np
from base_agent import BaseLinBandit
from utils import randargmax


class OFUL(BaseLinBandit):
    """
    OFUL: argmax_a  theta_hat . phi_a + sqrt(beta_t) * ||phi_a||_{V_t^{-1}}

    Noise model: R-subgaussian with known upper bound sigma_upper.
    Features and rewards are NOT sigma-scaled (homoscedastic OLS).
    beta uses the fixed sigma_upper, so lambda should be set to sigma_upper^2 / S^2
    to balance the two terms of the confidence radius.

    Per-arm sigma_vec is accepted for interface compatibility but only its max
    is used to adaptively tighten sigma when it is below sigma_upper.
    """

    def __init__(self, d, lam, delta, S=1.0, sigma_upper=1.0, baseline_arm=0):
        super().__init__(d, lam, delta, S, baseline_arm)
        self.sigma_upper = sigma_upper
        self._set_sigma(sigma_upper)   # initialize beta with the known bound

    def select_arm(self, features_h, sigma_vec, rng=None):
        # # Use min(sigma_upper, observed max) — never exceed the declared bound
        A = features_h.shape[0]
        scores = np.array([
            features_h[a] @ self.theta_hat
            + np.sqrt(self.beta_t * max(float(features_h[a] @ self.invVt @ features_h[a]), 0.0))
            for a in range(A)
        ])
        return int(randargmax(scores, rng))