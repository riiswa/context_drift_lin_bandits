import numpy as np
from base_agent import BaseLinBandit
from utils import randargmax


class LinTS(BaseLinBandit):
    """
    Linear Thompson Sampling (frequentist / confidence-set version).

    Sampling rule
    -------------
        eta          ~ N(0, I_d)
        theta_tilde   = theta_hat + beta_t^{1/2} * invVt^{1/2} @ eta
        A_t           = argmax_a  phi_a^T theta_tilde

    beta_t is the squared confidence radius inherited from BaseLinBandit,
    so sqrt(beta_t) is the scalar that inflates the ellipsoid — matching
    the OFUL radius and ensuring the sample stays in the confidence set
    in expectation.

    invVt^{1/2} is recomputed from the current invVt (updated via
    Sherman-Morrison in BaseLinBandit._ridge_update) after every update.

    Parameters
    ----------
    d            : feature dimension
    lam          : regularisation  (set to sigma^2 / S^2 for balance)
    delta        : confidence level
    S            : bound on ||theta*||_2
    sigma_upper  : sub-Gaussian noise upper bound (initialises beta_t)
    baseline_arm : fallback arm (interface compatibility)
    """

    def __init__(self, d, lam, delta, S=1.0, sigma_upper=1.0, baseline_arm=0):
        super().__init__(d, lam, delta, S, baseline_arm)
        self.sigma_upper  = sigma_upper
        self._set_sigma(sigma_upper)

        # V_0 = lam*I  =>  invV_0 = I/lam  =>  invV_0^{1/2} = I/sqrt(lam)
        self._invVt_sqrt = np.eye(d) / np.sqrt(lam)

    # ------------------------------------------------------------------
    # Episode interface (stationary: no drift handling needed)
    # ------------------------------------------------------------------

    def start_episode(self, features, sigma_mat):
        """Inherited no-op — LinTS has no per-episode state."""
        pass

    # ------------------------------------------------------------------
    # Arm selection
    # ------------------------------------------------------------------

    def select_arm(self, features_h, sigma_vec, rng=None):
        """
        Parameters
        ----------
        features_h : (A, d)  unscaled feature vectors for user h
        sigma_vec  : (A,)    per-arm noise std (accepted for interface
                             compatibility; not used in standard LinTS)
        rng        : np.random.Generator (optional)

        Returns
        -------
        int  index of selected arm
        """
        _rng = rng if rng is not None else np.random.default_rng()

        eta           = _rng.standard_normal(self.d)           # (d,)
        theta_tilde   = self.theta_hat + np.sqrt(self.beta_t) * (self._invVt_sqrt @ eta)

        scores = features_h @ theta_tilde                      # (A,)
        return int(randargmax(scores, rng))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, phi, r_tilde, arm, h, sigma_val):
        """
        Delegates the ridge update (Sherman-Morrison on invVt, beta_t
        refresh) to BaseLinBandit, then refreshes invVt^{1/2}.

        Parameters
        ----------
        phi       : (d,)   unscaled feature vector of chosen arm
        r_tilde   : float  observed reward
        arm       : int    chosen arm index
        h         : int    user index within episode
        sigma_val : float  noise std for chosen arm (not used here;
                           LinTS uses the fixed sigma_upper for beta_t)
        """
        super().update(phi, r_tilde, arm, h, sigma_val)
        # invVt was just updated by Sherman-Morrison — refresh its sqrt
        self._refresh_invVt_sqrt()

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    def _refresh_invVt_sqrt(self):
        """
        Compute invVt^{1/2} via symmetric eigendecomposition.

        invVt is symmetric PSD, so:
            invVt  = Q diag(v) Q^T   (eigh guarantees real, sorted eigenvalues)
          invVt^{1/2} = Q diag(sqrt(v)) Q^T

        np.linalg.eigh is used (not eig) for stability on symmetric matrices.
        Negative eigenvalues from floating-point drift are clamped to zero.
        """
        eigvals, eigvecs = np.linalg.eigh(self.invVt)
        eigvals          = np.maximum(eigvals, 0.0)
        self._invVt_sqrt = eigvecs * np.sqrt(eigvals) @ eigvecs.T