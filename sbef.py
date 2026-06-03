import numpy as np
from scipy.linalg import cho_factor, cho_solve

class SparseBayesExpander:
    """Sparse Bayesian image expansion filter (variational ARD) for one channel.

    Model: x = W y + eps, where x in R^D, y in R^Q. W is D x Q.
    We learn posterior q(W) ~ N(M, {Sigma_d}) and q(alpha) Gamma, q(beta) Gamma
    using the variational updates in Kanemura et al. (2009).
    """
    def __init__(self, D, Q, a_alpha0=20.0, b_alpha0=1e-6, a_beta0=1e-6, b_beta0=1e-6,
                 alpha_threshold=1e20, max_iter=200, tol=1e-6, verbose=False):
        self.D = D
        self.Q = Q
        self.a_alpha0 = a_alpha0
        self.b_alpha0 = b_alpha0
        self.a_beta0 = a_beta0
        self.b_beta0 = b_beta0
        self.alpha_threshold = alpha_threshold
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # parameters to learn
        # parameters to learn
        # self.M corresponds to the posterior mean of W (μ in the paper)
        # W is D x Q (D = r*r high-res pixels, Q = m*m low-res values)
        # μ_d (row d) predicts the d-th pixel of the high-res patch as μ_d^T y
        self.M = np.zeros((D, Q))  # posterior mean of W (initialized to 0)

        # self.Sigmas is a list of posterior covariances for each row d of W:
        # Sigma_d = Cov[w_d] (Q x Q). In the paper Σ_d = (Diag(E[α_d]) + E[β] S_yy)^{-1}.
        # We store one QxQ matrix per output dimension d.
        self.Sigmas = [np.eye(Q) for _ in range(D)]  # posterior covariances per row

        # hyperparameters (posterior parameters for the Gamma distributions)
        # For α (ARD per-weight) we maintain posterior Gamma(a_alpha, b_alpha)
        # where E[α] = a_alpha / b_alpha. Shapes: D x Q (one α per weight W_{d,q}).
        # We initialize a_alpha so the first update (a_alpha0 + 0.5) is consistent.
        self.a_alpha = np.ones((D, Q)) * (self.a_alpha0 + 0.5)
        # Add small epsilon to avoid division-by-zero when computing Ealpha = a/b
        self.b_alpha = np.ones((D, Q)) * (self.b_alpha0 + 1e-8)

        # For β (noise precision) we keep scalar Gamma(a_beta, b_beta) and
        # update a_beta = a_beta0 + 0.5 * N * D, b_beta uses residuals.
        self.a_beta = self.a_beta0
        self.b_beta = self.b_beta0

    def fit(self, Y, X):
        """Train the model.

        Y: Q x N low-res patches
        X: D x N high-res patches
        """
        Q, N = Y.shape
        D, N2 = X.shape
        assert Q == self.Q and D == self.D and N == N2

        # precompute
        S_yy = Y @ Y.T  # Q x Q
        Ty = [X[d, :] @ Y.T for d in range(D)]  # each is 1 x Q

        # initialize alpha and beta expectations
        Ealpha = self.a_alpha / self.b_alpha
        # init beta from data variance
        self.a_beta = self.a_beta0 + N * D / 2.0
        self.b_beta = self.b_beta0 + 0.5 * np.sum((X - 0)**2)
        E_beta = self.a_beta / self.b_beta

        prev_M = self.M.copy()

        for it in range(self.max_iter):
            # update q(A): a_alpha and b_alpha
            # requires <w^2> = mu^2 + diag(Sigma)
            for d in range(D):
                mu_d = self.M[d]
                Sigma_d = self.Sigmas[d]
                w2 = mu_d**2 + np.diag(Sigma_d)
                self.b_alpha[d, :] = self.b_alpha0 + 0.5 * w2
                self.a_alpha[d, :] = self.a_alpha0 + 0.5
            Ealpha = self.a_alpha / self.b_alpha

            # threshold large alphas (prune)
            Ealpha = np.where(Ealpha > self.alpha_threshold, np.inf, Ealpha)

            # update q(W): for each d, Sigma_d = (diag(Ealpha_d) + E_beta * S_yy)^-1
            E_beta = self.a_beta / self.b_beta
            for d in range(D):
                A_d = np.diag(Ealpha[d]) + E_beta * S_yy
                # use cho factorization for stability
                # try:
                #     c, low = cho_factor(A_d, check_finite=False)
                #     Sigma_d = cho_solve((c, low), np.eye(self.Q), check_finite=False)
                # except np.linalg.LinAlgError:
                Sigma_d = np.linalg.pinv(A_d)
                self.Sigmas[d] = Sigma_d
                # mu_d = E_beta * Sigma_d * (sum_n x_dn y_n) = E_beta * Sigma_d * Ty[d].T
                self.M[d] = E_beta * (Sigma_d @ Ty[d].T)

            # update q(beta)
            # need sum_n <|| x_n - W y_n ||^2>
            # Vectorized computation:
            #   sum_n ||x_n - M y_n||^2 = ||X - M Y||_F^2
            #   sum_n tr(y_n y_n^T * sum_d Sigma_d) = tr(Ssum * (Y Y^T))
            Ssum = np.zeros((self.Q, self.Q))
            for d in range(D):
                Ssum += self.Sigmas[d]
            # residual part
            residual = X - (self.M @ Y)
            sum_sq = np.sum(residual**2)
            # trace part using precomputed S_yy (Y @ Y.T)
            trace_part = float(np.trace(Ssum @ S_yy))
            sum_err = sum_sq + trace_part
            self.b_beta = self.b_beta0 + 0.5 * sum_err
            self.a_beta = self.a_beta0 + 0.5 * N * D
            E_beta = self.a_beta / self.b_beta

            # convergence check
            normM = np.linalg.norm(self.M, 'fro')
            delta = np.linalg.norm(self.M - prev_M, 'fro') / max(1e-12, normM)
            if self.verbose:
                print(f'it={it} delta={delta:.3e} E_beta={E_beta:.3e}')
            if self.verbose:
                print(f'it={it} delta={delta:.3e} E_beta={E_beta:.3e}')
            if delta < self.tol:
                stop_reason = f'converged (delta {delta:.3e} < tol {self.tol})'
                prev_M = self.M.copy()
                break
            prev_M = self.M.copy()
            stop_reason = 'max_iter'
        if self.verbose:
            try:
                print(f'Training stopped at it={it} reason={stop_reason}')
            except Exception:
                pass
        return self

    def transform_patch(self, y):
        """Apply learned filter to a single low-res patch y (Q, ) -> x (D, )"""
        return (self.M @ y).reshape(self.D)
