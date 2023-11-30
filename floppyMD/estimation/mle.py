"""
The code in this file was originnaly adapted from pymle (https://github.com/jkirkby3/pymle)
"""

import numpy as np
from scipy.optimize import minimize

from ..base import Estimator


class EstimatedResult(object):
    def __init__(self, params: np.ndarray, log_like: float, sample_size: int):
        """
        Container for the result of estimation
        :param params: array, the estimated (optimal) params
        :param log_like: float, the final log-likelihood value (at optimum)
        :param sample_size: int, the size of sample used in estimation (don't include S0)
        """
        self.params = params
        self.log_like = log_like
        self.sample_size = sample_size

    @property
    def likelihood(self) -> float:
        """The likelihood with estimated params"""
        return np.exp(self.log_like)

    @property
    def aic(self) -> float:
        """The AIC (Aikake Information Criteria) with estimated params"""
        return 2 * (len(self.params) - self.log_like)

    @property
    def bic(self) -> float:
        """The BIC (Bayesian Information Criteria) with estimated params"""
        return len(self.params) * np.log(self.sample_size) - 2 * self.log_like

    def __str__(self):
        """String representation of the class (for pretty printing the results)"""
        return f"\nparams      | {self.params} \n" f"sample size | {self.sample_size} \n" f"likelihood  | {self.log_like} \n" f"AIC         | {self.aic}\n" f"BIC         | {self.bic}"


class CallbackFunctor:
    """
    Callback or scipy minimize in order to store history if wanted
    """

    def __init__(self, obj_fun):
        """
        obj_fun is a provided function to extract value from OptimizedResult
        """
        self.history = [np.inf]
        self.sols = []
        self.num_calls = 0
        self.obj_fun = obj_fun

    def __call__(self, x):
        fun_val = self.obj_fun(x)
        self.num_calls += 1
        if fun_val < self.history[-1]:
            self.sols.append(x)
            self.history.append(fun_val)

    def save_sols(self, filename):
        sols = np.array([sol for sol in self.sols])
        np.savetxt(filename, sols)


class LikelihoodEstimator(Estimator):
    r"""Base class of all estimators

    Parameters
    ----------
    model : Model, optional, default=None
        A model which can be used for initialization. In case an estimator is capable of online learning, i.e.,
        capable of updating models, this can be used to resume the estimation process.
    """

    def __init__(self, transition, **kwargs):
        super().__init__(transition.model)
        self.transition = transition

    def fit(self, data, minimizer=None, params0=None, use_jac=True, **kwargs):
        r"""Fits data to the estimator's internal :class:`Model` and overwrites it. This way, every call to
        :meth:`fetch_model` yields an autonomous model instance. Sometimes a :code:`partial_fit` method is available,
        in which case the model can get updated by the estimator.

        Parameters
        ----------
        data : array_like
            Data that is used to fit a model.
        **kwargs
            Additional kwargs.

        Returns
        -------
        self : Estimator
            Reference to self.
        """

        if self.transition.do_preprocess_traj:
            for i, trj in enumerate(data):
                self.transition.preprocess_traj(trj)

        if minimizer is None:
            minimizer = minimize
        if params0 is None:
            raise NotImplementedError  # We could use then an exact estimation
        if self.transition.has_jac and use_jac:
            res = minimizer(self._log_likelihood_negative_with_jac, params0, args=(data,), jac=True, method="L-BFGS-B")
        else:
            res = minimizer(self._log_likelihood_negative, params0, args=(data,), method="L-BFGS-B")
        params = res.x

        final_like = -res.fun

        self.model.params = params
        self.model.fitted_ = True

        self.results_ = EstimatedResult(params=params, log_like=final_like, sample_size=data.nobs - 1)

        return self

    def _log_likelihood_negative(self, params, data, **kwargs):
        return self._loop_over_trajs(self.transition, data.weights, data, params, **kwargs)[0]

    def _log_likelihood_negative_with_jac(self, params, data, **kwargs):
        return self._loop_over_trajs(self.transition, data.weights, data, params, **kwargs)


class ELBOEstimator(LikelihoodEstimator):
    """
    Maximize the Evidence lower bound.
    Similar to EM estimation but the expectation is realized inside the minimization loop
    """

    def __init__(self, model=None, transition=None, **kwargs):
        super().__init__(model)


class EMEstimator(LikelihoodEstimator):
    """
    Maximize the likelihood using Expectation-maximization algorithm
    """

    def __init__(self, model, transition, *args, **kwargs):
        super().__init__(model)

    def fit():
        """
        In this do a loop that alternatively minimize and compute expectation
        """
