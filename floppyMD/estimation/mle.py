import numpy as np
from scipy.optimize import minimize

from ..base import Estimator


class LikelihoodEstimator(Estimator):
    r"""Base class of all estimators

    Parameters
    ----------
    model : Model, optional, default=None
        A model which can be used for initialization. In case an estimator is capable of online learning, i.e.,
        capable of updating models, this can be used to resume the estimation process.
    """

    def __init__(self, model=None):
        super().__init__(model)
        # Should check is the model is linear in parameters

    def fit(self, data, minimizer=None, **kwargs):
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
        if minimizer is None:
            minimizer = minimize
        res = minimizer(self.likelihood, bounds=self._param_bounds, guess=params0)
        params = res.params

        final_like = -res.value

        self.model.params = params

        return self


class LikelihoodFromTransitionProbability(LikelihoodEstimator):
    def __init__(self, model=None, transition=None):
        super().__init__(model)
