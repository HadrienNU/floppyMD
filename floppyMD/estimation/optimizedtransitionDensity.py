import numpy as np
import numba as nb

from .transitionDensity import TransitionDensity
from ..models.piecewiseOverdamped import linear_interpolation_with_gradient


class ExactDensity(TransitionDensity):
    def __init__(self, model):
        """
        Class which represents the exact transition density for a model (when available)
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    @property
    def do_preprocess_traj(self):
        return True

    def preprocess_traj(self, q, use_midpoint=False):
        """Preprocess colvar trajectory with a given grid for faster model optimization

        Args:
            q (list of ndarray): trajectories of the CV.
            knots (ndarray): CV values forming the knots of the piecewise-linear approximation of logD and gradF.

        Returns:
            traj (numba types list): list of tuples (bin indices, bin positions, displacements)
        """

        # TODO: enable subsampling by *averaging* biasing force in interval
        # Then run inputting higher-res trajectories

        traj = list()

        for qi in q:
            deltaq = qi[1:] - qi[:-1]

            if use_midpoint:
                # Use mid point of each interval
                # Implies a "leapfrog-style" integrator that is not really used for overdamped LE
                ref_q = 0.5 * (qi[:-1] + qi[1:])
            else:
                # Truncate last traj point to match deltaq array
                ref_q = qi[:-1]

            # bin index on possibly irregular grid
            idx = np.searchsorted(self.model.knots, ref_q)

            assert (idx > 0).all() and (idx < len(self.model.knots)).all(), "Out-of-bounds point(s) in trajectory\n"
            # # Other option: fold back out-of-bounds points - introduces biases
            # idx = np.where(idx == 0, 1, idx)
            # idx = np.where(idx == len(knots), len(knots) - 1, idx)

            q0, q1 = self.model.knots[idx - 1], self.model.knots[idx]
            # fractional position within the bin
            h = (qi[:-1] - q0) / (q1 - q0)

            traj.append((idx, h, deltaq))

        # Numba prefers typed lists
        return nb.typed.List(traj)

    def __call__(self, params, trj, dt):
        """
        The exact transition density (when applicable)
        Note: this will raise exception if the model does not implement exact_density
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        return objective_order1_debiased(params, self.model.knots, trj, dt, self.model.beta)


@nb.njit(parallel=True)
def objective_order1_debiased(params, knots, traj, dt, beta, f):
    """Objective function: order-1 OptLE for overdamped Langevin, order-1 propagator
    Includes the debiasing feature of Hallegot, Pietrucci and Hénin for time-dependent biases

    Args:
        params (ndarray): parameters of the model - piecewise-linear grad F (free energy) and log D
        knots (ndarray): CV values forming the knots of the piecewise-linear approximation of logD and gradF
        q (list of ndarray): trajectories of the CV
        deltaq (list of ndarray): trajectories of CV differences
        f (list of ndarray): trajectories of the biasing force

    Returns:
        real, ndarray: objective function and its derivatives with respect to model parameters
    """

    idx, h, deltaq = traj[i]
    G, logD, dXdk = linear_interpolation_with_gradient(idx, h, knots, params)
    # dXdk is the gradient with respect to the knots (same for all quantities)

    # Debiasing (truncate last traj point)
    # G -= f[i][:-1]

    phi = -beta * np.exp(logD) * G * dt
    dphidlD = -beta * np.exp(logD) * G * dt
    dphidG = -beta * np.exp(logD) * dt

    mu = 2.0 * np.exp(logD) * dt
    dmudlD = 2.0 * np.exp(logD) * dt
    logL = (0.5 * logD + np.square(deltaq - phi) / (2.0 * mu)).sum()
    dLdlD = 0.5 + (2.0 * (deltaq - phi) * -1.0 * dphidlD * (2.0 * mu) - np.square(deltaq - phi) * 2.0 * dmudlD) / np.square(2.0 * mu)
    dLdG = 2.0 * (deltaq - phi) * -1.0 * dphidG / (2.0 * mu)

    dlogLdkG = np.dot(dXdk, dLdG)
    dlogLdklD = np.dot(dXdk, dLdlD)

    return logL, np.concatenate((dlogLdkG, dlogLdklD))
