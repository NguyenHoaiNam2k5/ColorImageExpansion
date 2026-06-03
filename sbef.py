import numpy as np
from scipy.linalg import cho_factor, cho_solve


class SparseBayesExpander:
    """Sparse Bayesian Expander with symmetry and tied-alpha support.

    Implements a variational ARD model for x = W y + eps (x in R^D, y in R^Q).

    Paper mapping (informal):
    - q(W_d) = N(M_d, Sigma_d) where M_d is row d of self.M and Sigma_d is self.Sigmas[d]
      (corresponds to posterior over W rows in the paper).
    - q(alpha) ~ Gamma(a_alpha, b_alpha) : controls per-weight (or per-group) precision.
    - q(beta) ~ Gamma(a_beta, b_beta) : noise precision.

    symmetry: 'none' | 'h' | 'v' | 'hv' groups indices of the m x m low-res patch.
    tie_alpha_across_rows: if True, the same alpha (or group-alpha) is shared across all D rows.
    """

    def __init__(self, D, Q, a_alpha0=20.0, b_alpha0=1e-6, a_beta0=1e-6, b_beta0=1e-6,
                 alpha_threshold=np.exp(20), max_iter=200, tol=1e-6, verbose=False,
                 symmetry='none', tie_alpha_across_rows=False):
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

        if symmetry not in ('none', 'h', 'v', 'hv'):
            raise ValueError("symmetry must be one of 'none','h','v','hv'")
        self.symmetry = symmetry
        self.tie_alpha_across_rows = bool(tie_alpha_across_rows)

        # Posterior parameters
        self.M = np.zeros((D, Q), dtype=float)  # posterior means (D x Q)
        self.Sigmas = [np.eye(Q, dtype=float) for _ in range(D)]  # list of QxQ covariances

        # Build symmetry groups (map q -> group id) when requested
        self.group_of_q = None
        self.groups = None
        if self.symmetry != 'none':
            m = int(round(np.sqrt(Q)))
            if m * m != Q:
                raise ValueError('Q must be a perfect square for symmetry')
            groups_map = {}
            for q in range(Q):
                i = q // m
                j = q % m
                s = {(i, j)}
                if 'h' in self.symmetry:
                    s.add((i, m - 1 - j))
                if 'v' in self.symmetry:
                    s.add((m - 1 - i, j))
                if self.symmetry == 'hv':
                    s.add((m - 1 - i, m - 1 - j))
                idxs = tuple(sorted(ii * m + jj for (ii, jj) in s))
                groups_map.setdefault(idxs, set()).add(q)
            # groups: list of lists of q indices that belong to the same symmetry group
            self.groups = [sorted(list(k)) for k in groups_map.keys()]
            self.group_of_q = np.empty(Q, dtype=int)
            for gid, qlist in enumerate(self.groups):
                for q in qlist:
                    self.group_of_q[q] = gid

        # Initialize alpha posterior parameters depending on tie/group config
        if self.tie_alpha_across_rows:
            if self.groups is not None:
                G = len(self.groups)
                self.a_alpha = np.ones(G, dtype=float) * (self.a_alpha0 + 0.5)
                self.b_alpha = np.ones(G, dtype=float) * (self.b_alpha0 + 1e-8)
            else:
                self.a_alpha = np.ones(Q, dtype=float) * (self.a_alpha0 + 0.5)
                self.b_alpha = np.ones(Q, dtype=float) * (self.b_alpha0 + 1e-8)
        else:
            if self.groups is not None:
                G = len(self.groups)
                self.a_alpha = np.ones((D, G), dtype=float) * (self.a_alpha0 + 0.5)
                self.b_alpha = np.ones((D, G), dtype=float) * (self.b_alpha0 + 1e-8)
            else:
                self.a_alpha = np.ones((D, Q), dtype=float) * (self.a_alpha0 + 0.5)
                self.b_alpha = np.ones((D, Q), dtype=float) * (self.b_alpha0 + 1e-8)

        # beta posterior params
        self.a_beta = self.a_beta0
        self.b_beta = self.b_beta0

    def fit(self, Y, X):
        """Train with Y (Q x N) low-res patches and X (D x N) high-res patches.

        Algorithm (high-level): iterate updates for q(W), q(alpha), q(beta).

        Paper-equation mapping (sketch):
        - Sigma_d = (Diag(E[alpha]_d) + E[beta] S_yy)^{-1}
        - mu_d = E[beta] Sigma_d Ty[d].T
        - a_alpha/b_alpha from expectations of w^2 over tied groups
        - a_beta/b_beta from residuals + trace(Ssum S_yy)
        """
        Q, N = Y.shape
        D, N2 = X.shape
        assert Q == self.Q and D == self.D and N == N2

        S_yy = Y @ Y.T
        Ty = [X[d, :] @ Y.T for d in range(D)]

        # initialize beta posterior
        self.a_beta = self.a_beta0 + 0.5 * N * D
        self.b_beta = self.b_beta0 + 0.5 * np.sum(X**2)

        prev_M = self.M.copy()

        for it in range(self.max_iter):
            # Expected squared weights: E[w^2] = mu^2 + diag(Sigma)
            W2 = np.zeros((D, Q), dtype=float)
            for d in range(D):
                W2[d, :] = self.M[d] ** 2 + np.diag(self.Sigmas[d])

            # Update alpha posterior parameters depending on grouping/ties
            if self.tie_alpha_across_rows:
                # Single alpha per group/q across rows
                if self.groups is not None:
                    G = len(self.groups)
                    a_new = np.empty(G, dtype=float)
                    b_new = np.empty(G, dtype=float)
                    for gid, qlist in enumerate(self.groups):
                        sum_w2 = np.sum(W2[:, qlist])  # sum over rows and q in group
                        n_weights = D * len(qlist)
                        b_new[gid] = self.b_alpha0 + 0.5 * sum_w2
                        a_new[gid] = self.a_alpha0 + 0.5 * n_weights
                    self.a_alpha = a_new
                    self.b_alpha = b_new
                    # expand to per-weight Ealpha (D x Q)
                    Ealpha = np.empty((D, Q), dtype=float)
                    for q in range(Q):
                        gid = self.group_of_q[q]
                        Ealpha[:, q] = self.a_alpha[gid] / self.b_alpha[gid]
                else:
                    # per-q shared across rows
                    sum_w2 = np.sum(W2, axis=0)
                    b_new = self.b_alpha0 + 0.5 * sum_w2
                    a_new = self.a_alpha0 + 0.5 * D
                    self.a_alpha = np.ones(Q, dtype=float) * a_new
                    self.b_alpha = b_new
                    Ealpha = np.tile(self.a_alpha / self.b_alpha, (D, 1))
            else:
                # Alphas not tied across rows
                if self.groups is not None:
                    G = len(self.groups)
                    a_new = np.empty((D, G), dtype=float)
                    b_new = np.empty((D, G), dtype=float)
                    for d in range(D):
                        for gid, qlist in enumerate(self.groups):
                            sum_w2 = np.sum(W2[d, qlist])
                            n_weights = len(qlist)
                            b_new[d, gid] = self.b_alpha0 + 0.5 * sum_w2
                            a_new[d, gid] = self.a_alpha0 + 0.5 * n_weights
                    # expand to per-weight arrays
                    self.a_alpha = np.zeros((D, Q), dtype=float)
                    self.b_alpha = np.zeros((D, Q), dtype=float)
                    Ealpha = np.empty((D, Q), dtype=float)
                    for q in range(Q):
                        gid = self.group_of_q[q]
                        self.a_alpha[:, q] = a_new[:, gid]
                        self.b_alpha[:, q] = b_new[:, gid]
                        Ealpha[:, q] = self.a_alpha[:, q] / self.b_alpha[:, q]
                else:
                    # fully independent per-weight
                    self.b_alpha = self.b_alpha0 + 0.5 * W2
                    self.a_alpha = self.a_alpha0 + 0.5
                    Ealpha = self.a_alpha / self.b_alpha

            # prune huge alphas for numeric stability
            Ealpha = np.where(Ealpha > self.alpha_threshold, np.inf, Ealpha)

            # Update q(W): per-row Gaussian
            E_beta = self.a_beta / self.b_beta
            for d in range(D):
                A_d = np.diag(Ealpha[d]) + E_beta * S_yy
                try:
                    c, low = cho_factor(A_d, check_finite=False)
                    Sigma_d = cho_solve((c, low), np.eye(Q), check_finite=False)
                except np.linalg.LinAlgError:
                    Sigma_d = np.linalg.pinv(A_d)
                self.Sigmas[d] = Sigma_d
                # mu_d = E_beta * Sigma_d @ Ty[d].T  (paper equation)
                self.M[d] = E_beta * (Sigma_d @ Ty[d].T)

            # Update q(beta)
            Ssum = np.zeros((Q, Q), dtype=float)
            for d in range(D):
                Ssum += self.Sigmas[d]
            residual = X - (self.M @ Y)
            sum_sq = np.sum(residual**2)
            trace_part = float(np.trace(Ssum @ S_yy))
            sum_err = sum_sq + trace_part
            self.b_beta = self.b_beta0 + 0.5 * sum_err
            self.a_beta = self.a_beta0 + 0.5 * N * D

            # convergence check
            normM = np.linalg.norm(self.M, 'fro')
            delta = np.linalg.norm(self.M - prev_M, 'fro') / max(1e-12, normM)
            if self.verbose:
                print(f'it={it} delta={delta:.3e} E_beta={(self.a_beta/self.b_beta):.3e}')
            if delta < self.tol:
                break
            prev_M = self.M.copy()

        return self

    def transform_patch(self, y):
        """Apply learned filter to a single low-res patch y (Q,) -> x (D,)."""
        return (self.M @ y).reshape(self.D)

