import pytest
import os
import numpy as np
import floppyMD
import dask.array as da


@pytest.fixture
def data(request):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/example_2d.trj"))
    if request.param == "dask":
        trj = da.from_array(trj)
    trj_list = floppyMD.Trajectories(dt=trj[1, 0] - trj[0, 0])
    for i in range(1, trj.shape[1]):
        trj_list.append(trj[:, 1:2])
    trj_list.stats
    return trj_list


@pytest.mark.parametrize(
    "data",
    [
        "numpy",
    ],
    indirect=True,
)
def test_numba_likelihood_estimator(data, request):
    n_knots = 20
    epsilon = 1e-10
    model = floppyMD.models.OverdampedFreeEnergy(np.linspace(data.stats.min - epsilon, data.stats.max + epsilon, n_knots), 1.0)
    estimator = floppyMD.LikelihoodEstimator(floppyMD.EulerNumbaOptimizedDensity(model))

    model = estimator.fit_fetch(data, coefficients0=np.concatenate((np.zeros(n_knots), np.zeros(n_knots) + 1.0)))
    assert model.fitted_
