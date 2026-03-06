import numpy as np
from base_agent import BaseLinBandit
from optimal_design import calc_q_opt_design


class LinMED(BaseLinBandit):
    """
    LinMED adapted to the H-user episodic setting.

    Samples from a distribution mixing:
      - G-optimal design weighted by MED scores  (opt_coeff)
      - Extra mass on the empirical best arm      (emp_coeff)
      - Uniform spread                            (1 - opt_coeff - emp_coeff)

    Fixes vs previous version
    -------------------------
    1. sigma / R not set:
       The reference uses R (= sqrt(lam)*S) as the noise constant in gamma_t /
       beta_t.  The previous code left self._sigma = 1.0 (BaseLinBandit default).
       Fix: call _set_sigma(R) once at __init__, frozen for the entire run.

    2. t double-increment:
       select_arm was incrementing self.t internally, while BaseLinBandit._ridge_update
       also increments self.t on every update call.  The reference only increments t
       in update_delayed / update.
       Fix: removed self.t += 1 from select_arm entirely; the base-class increment
       is sufficient.  The t == 1 guard still fires correctly on the very first call.

    3. G-optimal design computed on raw features instead of MED-scaled features:
       The reference scales each row of X_t by sqrt(MED_quo[i]) before passing to
       calc_q_opt_design (the AugX block).  The previous code passed unscaled
       features_h to _approx_design and then multiplied by weights separately,
       which is not equivalent.
       Fix: build AugX = features_h * sqrt(weights)[:, None] and pass that to
       calc_q_opt_design, exactly as the reference does.
    """

    def __init__(self, d, lam, delta, S=1.0,
                 opt_coeff=0.99, emp_coeff=0.005, baseline_arm=0):
        super().__init__(d, lam, delta, S, baseline_arm)
        self.opt_coeff  = opt_coeff
        self.emp_coeff  = emp_coeff
        self.each_coeff = 1.0 - opt_coeff - emp_coeff

        # FIX 1: set sigma = R = sqrt(lam)*S, frozen for the entire run,
        # matching the reference which uses R explicitly in calc_sqrt_beta_det2.
        R = float(np.sqrt(lam) * S)
        self._set_sigma(R)

    def _approx_design(self, features):
        """Return normalised G-optimal design probabilities for given feature matrix."""
        prob = calc_q_opt_design(features)   # shape (A, 1) or (A,)
        prob = np.asarray(prob).flatten()
        s = prob.sum()
        return prob / s if s > 1e-12 else np.ones(len(prob)) / len(prob)

    def _calc_med_weights(self, features_h, gap, best):
        """
        MED weights using the difference-vector leverage score:
            w_a = exp( -gap_a^2 / (beta_t * ||(a - a_best)||^2_{V^{-1}} ) )

        If the difference leverage score is zero (a == a_best or collinear),
        the weight is set to 1, matching Lin_SGMED's branch:
            if vVal_lev_score_a != 0: ... else: MED_quo[i] = 1
        """
        A      = features_h.shape[0]
        a_best = features_h[best]
        weights = np.zeros(A)
        for a in range(A):
            diff     = features_h[a] - a_best
            lev_diff = float(diff @ self.invVt @ diff)
            if lev_diff == 0.0:
                weights[a] = 1.0                          # collinear / best arm itself
            else:
                denom      = max(self.beta_t * lev_diff, 1e-12)
                weights[a] = np.exp(-gap[a] ** 2 / denom)
        return weights

    def select_arm(self, features_h, sigma_vec, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        A     = features_h.shape[0]
        means = features_h @ self.theta_hat
        best  = int(np.argmax(means))
        gap   = means[best] - means          # (A,) >= 0

        # ---- t == 1 special case (matches Lin_SGMED) -------------------------
        # theta_hat is still zero; skip MED weighting and use only opt design
        # with uniform spread, no empirical-best bonus.
        if self.t == 1:
            q_opt = self._approx_design(features_h)
            prob  = self.opt_coeff * q_opt + (1.0 - self.opt_coeff) / A
            prob  = prob / prob.sum()
            # FIX 2: do NOT increment self.t here — _ridge_update handles it.
            return int(rng.choice(A, p=prob))

        # ---- t > 1: full MED mixture -----------------------------------------
        weights = self._calc_med_weights(features_h, gap, best)

        # FIX 3: compute G-optimal design on MED-scaled features (AugX),
        # matching the reference:
        #   for i in range(K):
        #       AugX[i, :] = np.sqrt(MED_quo[i][0]) * X_t[i, :]
        #   prob_dist = calc_q_opt_design(AugX)
        aug_features = features_h * np.sqrt(weights)[:, None]
        q_opt        = self._approx_design(aug_features)

        qt        = self.opt_coeff * q_opt + self.each_coeff / A
        qt[best] += self.emp_coeff             # extra mass on empirical best

        prob = qt * weights
        s    = prob.sum()
        prob = prob / s if s > 1e-12 else np.ones(A) / A

        # FIX 2: do NOT increment self.t here — _ridge_update handles it.
        return int(rng.choice(A, p=prob))