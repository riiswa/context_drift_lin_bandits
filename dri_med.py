import numpy as np

from optimal_design import calc_q_opt_design
from dri_imed import DRI_IMED


class DRI_MED(DRI_IMED):
    """
    Dri-MED: Algorithm 1 with MED stochastic arm selection (MED step algorithm).

    Inherits all of Dri-IMED: episode-start discount, tau, nu/update logic.
    Only select_arm is replaced with the stochastic MED policy.
    """

    def __init__(self, d, lam, delta, S=1.0, nu0=1.0, epsilon=0.1,
                 gamma_decay=0.99, pi0=None,
                 opt_coeff=0.99, emp_coeff=0.005):
        super().__init__(d=d, lam=lam, delta=delta, S=S, nu0=nu0, epsilon=epsilon,
                         gamma_decay=gamma_decay, pi0=pi0)
        self.opt_coeff  = opt_coeff
        self.emp_coeff  = emp_coeff
        self.each_coeff = 1.0 - opt_coeff - emp_coeff
        self.t = 1

    def _approx_design(self, features_h):
        prob = calc_q_opt_design(features_h)  # returns shape (A, 1)
        return prob.flatten()  # reshape to (A,) for your code

    def select_arm(self, features_h, sigma_vec, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        features_s = features_h / np.maximum(sigma_vec, 1e-12)[:, None]
        A = features_s.shape[0]

        # Line 1: beta, LCB
        beta = self._compute_beta()
        sqrt_beta = np.sqrt(beta)
        means = features_s @ self.theta_hat
        best = int(np.argmax(means))

        lcb = np.array([
            means[a] - sqrt_beta * np.sqrt(max(float(features_s[a] @ self.invVt @ features_s[a]), 0.0))
            for a in range(A)
        ])

        # Lines 3-4: f_h, f_tilde_h
        f = np.array([
            np.exp(-(means[best] - means[a]) ** 2 /
                   max(beta * float(
                       (features_s[best] - features_s[a]) @ self.invVt @ (features_s[best] - features_s[a])), 1e-300))
            for a in range(A)
        ])

        penalties = self.nu * np.maximum(0.0, self.tau - lcb)
        penalties -= penalties.min()  # shift so best arm has penalty=0
        f_tilde = f * np.exp(-penalties)

        # Lines 5-6: ApproxDesign on sqrt(f_tilde)-scaled arms
        scaled = np.sqrt(f_tilde)[:, None] * features_s
        q_opt = self._approx_design(scaled)

        # Line 7: mix
        q = self.opt_coeff * q_opt + self.each_coeff / A
        q[best] += self.emp_coeff

        # Line 8: p_h'
        p_prime = q * f_tilde
        p_prime = p_prime / p_prime.sum()

        # Lines 9-10: B_h boost
        B_h = [a for a in range(A)
               if float(features_s[a] @ self.invVt @ features_s[a]) > 1.0]
        if B_h:
            p_final = 0.5 * p_prime
            p_final[B_h] += 0.5 / len(B_h)
        else:
            p_final = p_prime

        return int(rng.choice(A, p=p_final))