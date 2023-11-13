from typing import Callable, Optional, Union
from abc import abstractmethod
import numpy as np

from ..base import Model


class ModelOverdamped(Model):
    def __init__(self, has_exact_density: bool = False):
        """
        Base model for overdamped Langevin equations, defined by

        dX(t) = mu(X,t)dt + sigma(X,t)dW_t

        :param has_exact_density: bool, set to true if an exact density is implemented
        """
        self._has_exact_density = has_exact_density
        self._params: Optional[np.ndarray] = None
        self.h = 1e-05

    @abstractmethod
    def force(self, x: Union[float, np.ndarray], t: Union[float, np.ndarray] = 0.0) -> Union[float, np.ndarray]:
        """The drift term of the model"""
        raise NotImplementedError

    @abstractmethod
    def diffusion(self, x: Union[float, np.ndarray], t: Union[float, np.ndarray] = 0.0) -> Union[float, np.ndarray]:
        """The diffusion term of the model"""
        raise NotImplementedError

    # ==============================
    # Exact Transition Density and Simulation Step, override when available
    # ==============================

    @property
    def has_exact_density(self) -> bool:
        """Return true if model has an exact density implemented"""
        return self._has_exact_density

    @property
    def is_linear(self) -> bool:
        """Return True is the model is linear in its parameters"""
        return False

    def exact_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        """
        In the case where the exact transition density,
        P(Xt, t | X0) is known, override this method
        :param x0: float, the current value
        :param xt: float, the value to transition to
        :param t0: float, the time of observing x0
        :param dt: float, the time step between x0 and xt
        :return: probability
        """
        raise NotImplementedError

    def AitSahalia_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        """
        In the case where the Ait-Sahalia density expansion is known for this particular model, return it,
            else raises exception
        :param x0: float, the current value
        :param xt: float, the value to transition to
        :param t0: float, the time of observing x0
        :param dt: float, the time of observing Xt
        :return: probability via Ait-Sahalia expansion
        """
        raise NotImplementedError

    def exact_step(self, t: float, dt: float, x: Union[float, np.ndarray], dZ: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Exact Simulation Step, Implement if known (e.g. Browian motion or GBM)"""
        raise NotImplementedError

    # ==============================
    # Derivatives (Numerical By Default)
    # ==============================

    def force_x(self, x: Union[float, np.ndarray], t: float = 0.0) -> Union[float, np.ndarray]:
        """Calculate first spatial derivative of drift, dmu/dx"""
        return (self.drift(x + self.h, t) - self.drift(x - self.h, t)) / (2 * self.h)

    def force_t(self, x: Union[float, np.ndarray], t: float = 0.0) -> Union[float, np.ndarray]:
        """Calculate first time derivative of drift, dmu/dt"""
        return (self.drift(x, t + self.h) - self.drift(x, t)) / self.h

    def force_xx(self, x: Union[float, np.ndarray], t: float = 0.0) -> Union[float, np.ndarray]:
        """Calculate second spatial derivative of drift, d^2mu/dx^2"""
        return (self.drift(x + self.h, t) - 2 * self.drift(x, t) + self.drift(x - self.h, t)) / (self.h * self.h)

    def diffusion_x(self, x: Union[float, np.ndarray], t: float = 0.0) -> Union[float, np.ndarray]:
        """Calculate first spatial derivative of diffusion term, dsigma/dx"""
        return (self.diffusion(x + self.h, t) - self.diffusion(x - self.h, t)) / (2 * self.h)

    def diffusion_xx(self, x: Union[float, np.ndarray], t: float = 0.0) -> Union[float, np.ndarray]:
        """Calculate second spatial derivative of diffusion term, d^2sigma/dx^2"""
        return (self.diffusion(x + self.h, t) - 2 * self.diffusion(x, t) + self.diffusion(x - self.h, t)) / (self.h * self.h)
