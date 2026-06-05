import numpy as np
from scipy.linalg import cho_factor, cho_solve


class SparseBayesExpander:
    """One-channel sparse Bayesian image-expansion filter.

    This class implements the variational ARD learning algorithm used by
    Kanemura, Maeda, and Ishii for the scalar-channel model

        x = W y + eps,

    where y is an m x m low-resolution patch flattened to length Q and x is an
    r x r high-resolution patch flattened to length D.  Color handling is done
    outside this class by fitting one model per RGB/YIQ channel.

    The paper's independent ARD precisions are alpha[d, q].  When ``symmetry``
    is enabled, this implementation ties alpha values only for coefficients
    that are related by the paper's horizontal/vertical symmetry over both the
    output sub-pixel index d and the input patch index q.  It does not tie a
    single alpha across all output rows.
    """

    def __init__(self, D, Q, a_alpha0=20.0, b_alpha0=2e-8,
                 a_beta0=1e-6, b_beta0=1e-6, alpha_threshold=np.exp(20),
                 max_iter=200, tol=1e-6, verbose=False, symmetry='hv',
                 tie_alpha_across_rows=None):
        self.D = int(D)
        self.Q = int(Q)
        self.a_alpha0 = float(a_alpha0)
        self.b_alpha0 = float(b_alpha0)
        self.a_beta0 = float(a_beta0)
        self.b_beta0 = float(b_beta0)
        self.alpha_threshold = float(alpha_threshold)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = bool(verbose)

        if symmetry not in ('none', 'h', 'v', 'hv'):
            raise ValueError("symmetry must be one of 'none', 'h', 'v', 'hv'")
        self.symmetry = symmetry

        # Kept only for backwards compatibility with older versions of this
        # educational repo.  The paper-symmetric implementation below ignores
        # row-wise tying because the paper ties alpha over symmetric (d, q)
        # pairs, not across all d for the same q.
        self.tie_alpha_across_rows = tie_alpha_across_rows

        self.M = np.zeros((self.D, self.Q), dtype=np.float64)
        self.Sigmas = [np.eye(self.Q, dtype=np.float64) for _ in range(self.D)]

        self.alpha_groups, self.pair_to_group = self._build_alpha_groups()
        self.n_alpha_groups = len(self.alpha_groups)
        self.a_alpha = np.full(self.n_alpha_groups, self.a_alpha0 + 0.5,
                               dtype=np.float64)
        self.b_alpha = np.full(self.n_alpha_groups, self.b_alpha0 + 1e-8,
                               dtype=np.float64)
        self.Ealpha = self._expand_group_values(self.a_alpha / self.b_alpha)
        self.pruned_mask = self.Ealpha > self.alpha_threshold

        self.a_beta = self.a_beta0
        self.b_beta = self.b_beta0
        self.n_iter_ = 0
        self.converged_ = False

    @staticmethod
    def _square_side(n, name):
        side = int(round(np.sqrt(n)))
        if side * side != n:
            raise ValueError(f'{name} must be a perfect square')
        return side

    def _symmetry_orbit(self, d, q):
        """Return all (d, q) pairs related by requested patch symmetries."""
        if self.symmetry == 'none':
            return {(d, q)}

        r = self._square_side(self.D, 'D')
        m = self._square_side(self.Q, 'Q')
        du, dv = divmod(d, r)
        qi, qj = divmod(q, m)

        transforms = [(False, False)]
        if 'h' in self.symmetry:
            transforms.append((True, False))
        if 'v' in self.symmetry:
            transforms.append((False, True))
        if self.symmetry == 'hv':
            transforms.append((True, True))

        orbit = set()
        for flip_h, flip_v in transforms:
            ndu = r - 1 - du if flip_v else du
            ndv = r - 1 - dv if flip_h else dv
            nqi = m - 1 - qi if flip_v else qi
            nqj = m - 1 - qj if flip_h else qj
            orbit.add((ndu * r + ndv, nqi * m + nqj))
        return orbit

    def _build_alpha_groups(self):
        """Build paper-style groups over coefficient pairs (d, q)."""
        groups_by_key = {}
        for d in range(self.D):
            for q in range(self.Q):
                key = tuple(sorted(self._symmetry_orbit(d, q)))
                groups_by_key.setdefault(key, list(key))

        groups = list(groups_by_key.values())
        pair_to_group = np.empty((self.D, self.Q), dtype=np.int64)
        for gid, pairs in enumerate(groups):
            for d, q in pairs:
                pair_to_group[d, q] = gid
        return groups, pair_to_group

    def _expand_group_values(self, group_values):
        out = np.empty((self.D, self.Q), dtype=np.float64)
        for gid, pairs in enumerate(self.alpha_groups):
            for d, q in pairs:
                out[d, q] = group_values[gid]
        return out

    def _update_alpha(self):
        """Update q(alpha) from E[w_dq^2] = mu_dq^2 + Sigma_d[qq]."""
        W2 = np.empty((self.D, self.Q), dtype=np.float64)
        for d in range(self.D):
            W2[d] = self.M[d] ** 2 + np.diag(self.Sigmas[d])

        a_new = np.empty(self.n_alpha_groups, dtype=np.float64)
        b_new = np.empty(self.n_alpha_groups, dtype=np.float64)
        for gid, pairs in enumerate(self.alpha_groups):
            sum_w2 = 0.0
            for d, q in pairs:
                sum_w2 += W2[d, q]
            n_weights = len(pairs)
            a_new[gid] = self.a_alpha0 + 0.5 * n_weights
            b_new[gid] = self.b_alpha0 + 0.5 * sum_w2

            # a_new[gid] = n_weights * self.a_alpha0 + 0.5 * n_weights
            # b_new[gid] = n_weights * self.b_alpha0 + 0.5 * sum_w2

            # a_new[gid] = n_weights * self.a_alpha0 - 0.5 * n_weights + 1.0
            # b_new[gid] = n_weights * self.b_alpha0 + 0.5 * sum_w2

        self.a_alpha = a_new
        self.b_alpha = b_new
        self.Ealpha = self._expand_group_values(self.a_alpha / self.b_alpha)
        self.pruned_mask = self.Ealpha > self.alpha_threshold

    def fit(self, Y, X):
        """Fit the variational sparse Bayesian filter.

        Parameters
        ----------
        Y : ndarray, shape (Q, N)
            Low-resolution patches.
        X : ndarray, shape (D, N)
            Corresponding high-resolution r x r patches.
        """
        Y = np.asarray(Y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        if Y.ndim != 2 or X.ndim != 2:
            raise ValueError('Y and X must be 2-D arrays with shapes (Q, N) and (D, N)')

        Q, N = Y.shape
        D, N2 = X.shape
        if Q != self.Q or D != self.D or N != N2:
            raise ValueError(
                f'expected Y shape ({self.Q}, N) and X shape ({self.D}, N); '
                f'got {Y.shape} and {X.shape}'
            )
        if N == 0:
            raise ValueError('empty training set')

        S_yy = Y @ Y.T
        Ty = X @ Y.T  # shape (D, Q), Ty[d] = sum_n x_nd y_n^T

        self.a_beta = self.a_beta0 + 0.5 * N * self.D
        self.b_beta = self.b_beta0 + 0.5 * np.sum(X ** 2)

        prev_M = self.M.copy()
        self.converged_ = False

        for it in range(self.max_iter):
            self._update_alpha()
            E_beta = self.a_beta / self.b_beta

            for d in range(self.D):
                active = np.isfinite(self.Ealpha[d]) & (self.Ealpha[d] <= self.alpha_threshold)
                Sigma_d = np.zeros((self.Q, self.Q), dtype=np.float64)
                mu_d = np.zeros(self.Q, dtype=np.float64)

                if np.any(active):
                    idx = np.flatnonzero(active)
                    A = (np.diag(self.Ealpha[d, idx]) +
                         E_beta * S_yy[np.ix_(idx, idx)])
                    rhs = Ty[d, idx]

                    try:
                        c, low = cho_factor(A, check_finite=False)
                        Sigma_active = cho_solve((c, low), np.eye(len(idx)), check_finite=False)
                        mu_active = E_beta * cho_solve((c, low), rhs, check_finite=False)
                    except np.linalg.LinAlgError:
                        Sigma_active = np.linalg.pinv(A)
                        mu_active = E_beta * (Sigma_active @ rhs)

                    Sigma_d[np.ix_(idx, idx)] = Sigma_active
                    mu_d[idx] = mu_active

                self.Sigmas[d] = Sigma_d
                self.M[d] = mu_d

            Ssum = np.zeros((self.Q, self.Q), dtype=np.float64)
            for Sigma_d in self.Sigmas:
                Ssum += Sigma_d

            residual = X - self.M @ Y
            sum_err = float(np.sum(residual ** 2) + np.trace(Ssum @ S_yy))
            self.b_beta = self.b_beta0 + 0.5 * sum_err
            self.a_beta = self.a_beta0 + 0.5 * N * self.D

            prev_norm_M = np.linalg.norm(prev_M, 'fro')
            delta = np.linalg.norm(self.M - prev_M, 'fro') / max(1e-12, prev_norm_M)
            self.n_iter_ = it + 1

            if self.verbose:
                active_count = int(np.sum(self.active_mask()))
                print(
                    f'it={it:03d} delta={delta:.3e} '
                    f'E_beta={(self.a_beta / self.b_beta):.3e} active={active_count}'
                )

            if delta < self.tol:
                self.converged_ = True
                break
            prev_M = self.M.copy()

        return self

    def transform_patch(self, y):
        """Apply the learned posterior-mean filter to one low-resolution patch."""
        y = np.asarray(y, dtype=np.float64).reshape(self.Q)
        return self.M @ y

    def active_mask(self):
        """Boolean mask of coefficients kept after the paper's exp(20) ARD cutoff."""
        return np.isfinite(self.Ealpha) & (self.Ealpha <= self.alpha_threshold)

    def support_union_mask(self):
        """Input-pixel support used by at least one output sub-pixel filter row."""
        return np.any(self.active_mask(), axis=0)

    def support_size(self, union=True):
        """Return active support size.

        union=True counts input pixels active in at least one output sub-pixel row.
        union=False counts all active coefficients over all rows.
        For Fig. 3-style support, use mean_row_support_size / paper_support_size.
        """
        if union:
            return int(np.sum(self.support_union_mask()))
        return int(np.sum(self.active_mask()))
