import numpy as np
from . import Basis
from ._data_describe import quick_describe

import scipy.interpolate


class SplineFctBasis(Basis):
    """
    A single basis function that is given from splines fit of data
    """

    def __init__(self, knots, coeffs, k=3, periodic=False):
        self.periodic = periodic
        self.k = k
        self.t = knots  # knots are position along the axis of the knots
        self.c = coeffs
        self.const_removed = False
        self.dim_out_basis = 1

    def fit(self, describe_result):
        if isinstance(describe_result, np.ndarray):
            describe_result = quick_describe(describe_result)
        self.spl_ = scipy.interpolate.BSpline(self.t, self.c, self.k)
        self.n_output_features_ = describe_result.mean.shape[0]
        return self

    def transform(self, X, **kwargs):
        return self.spl_(X)

    def deriv(self, X, deriv_order=1):
        nsamples, dim = X.shape
        grad = np.zeros((nsamples, dim) + (dim,) * deriv_order)
        for i in range(dim):
            grad[(Ellipsis, slice(i, i + 1)) + (i,) * (deriv_order)] = self.spl_.derivative(deriv_order)(X[:, slice(i, i + 1)])
        return grad

    def hessian(self, X):
        return self.deriv(X, deriv_order=2)

    def antiderivative(self, X, order=1):
        return self.spl_.antiderivative(order)(X)
