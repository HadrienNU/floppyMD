from .overdamped import ModelOverdamped
import numpy as np
import numba as nb


@nb.njit
def linear_interpolation_with_gradient(idx, h, knots, fp):
    n_knots = knots.shape[0]
    # x0, x1 = knots[idx - 1], knots[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    # Second parameter set is in second half of array
    g0, g1 = fp[idx - 1 + n_knots], fp[idx + n_knots]

    hm = 1 - h
    val_f = hm * f0 + h * f1
    val_g = hm * g0 + h * g1

    # Set gradient elements one by one
    grad = np.zeros((knots.shape[0], idx.shape[0]))
    for i, ik in enumerate(idx):
        grad[ik - 1, i] = hm[i]
        grad[ik, i] = h[i]
    return val_f, val_g, grad


class OverdampedFreeEnergy(ModelOverdamped):
    """
    TODO: A class that implement a overdamped model with a given free energy
    """

    def __init__(self, knots, beta, **kwargs):
        super().__init__()
        self.knots = knots
        self.beta = beta
        self._size_basis = self.basis.n_output_features_

    def force(self, x, t: float = 0.0):
        G, logD, dXdk = linear_interpolation_with_gradient(idx, h, self.knots, self._params)
        return -self.beta * np.exp(logD) * G

    def diffusion(self, x, t: float = 0.0):
        G, logD, dXdk = linear_interpolation_with_gradient(idx, h, self.knots, self._params)
        return 2.0 * np.exp(logD)

    def force_t(self, x, t: float = 0.0):
        return 0.0

    def force_x(self, x, t: float = 0.0):
        return np.dot(self._params[: self._size_basis], self.basis.derivative(x))

    def force_xx(self, x, t: float = 0.0):
        return np.dot(self._params[: self._size_basis], self.basis.hessian(x))

    def diffusion_t(self, x, t: float = 0.0):
        return 0.0

    def diffusion_x(self, x, t: float = 0.0):
        return np.dot(self._params[self._size_basis :], self.basis.derivative(x))

    def diffusion_xx(self, x, t: float = 0.0):
        return np.dot(self._params[self._size_basis :], self.basis.hessian(x))

    def is_linear(self):
        return True
