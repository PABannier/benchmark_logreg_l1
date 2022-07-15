from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import warnings
    import numpy as np
    from scipy.sparse import issparse
    # from skglm.estimators import SparseLogisticRegression
    from skglm.datafits import Logistic
    from skglm.penalties import L1
    from skglm.solvers.cd_solver import cd_solver
    from skglm.solvers.prox_newton_solver import prox_newton_solver
    # from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = "skglm"
    stopping_strategy = "iteration"

    install_cmd = 'conda'
    requirements = [
        'pip:skglm'
    ]
    references = [
        'Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel'
        'and M. Massias'
        '"Beyond L1: Faster and Better Sparse Models with skglm", '
        'https://arxiv.org/abs/2204.07826'
    ]

    parameters = {
        "solver": ["prox_newton", "cd"],
        "cst_step_size": [True, False]}

    def skip(self, X, y, lmbd):
        if issparse(X):
            return True, "No comparison with sparse matrices"
        return False, None

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        n_samples = self.X.shape[0]

        self.datafit = Logistic()
        if issparse(X):
            self.datafit.initialize_sparse(X.data, X.indptr, X.indices, y)
        else:
            self.datafit.initialize(X, y)

        self.penalty = L1(lmbd / n_samples)

        # warnings.filterwarnings('ignore', category=ConvergenceWarning)
        # self.logreg = SparseLogisticRegression(
        #     alpha=self.lmbd / n_samples, max_iter=1, max_epochs=50_000,
        #     tol=1e-12, fit_intercept=False, warm_start=False, verbose=False)

        # Cache Numba compilation
        self.run(1)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros(self.X.shape[1])
        else:

            w = np.zeros(self.X.shape[1])
            Xw = np.zeros(self.X.shape[0])

            if self.solver == 'prox_newton':
                coef = prox_newton_solver(
                    self.X, self.y, self.datafit, self.penalty, w, Xw,
                    max_iter=n_iter, tol=1e-12, verbose=0,
                    cst_step_size=self.cst_step_size)[0]
            else:
                coef = cd_solver(
                    self.X, self.y, self.datafit, self.penalty, w, Xw, max_iter=n_iter,
                    tol=1e-12)[0]

            self.coef = coef

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def get_result(self):
        return self.coef
