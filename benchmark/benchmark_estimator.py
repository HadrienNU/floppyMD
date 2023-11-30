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


@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
def test_likelihood_optimization(data, request, benchmark):
    bf = floppyMD.function_basis.Linear().fit(data)
    model = floppyMD.models.OverdampedBF(bf)
    estimator = floppyMD.LikelihoodEstimator(floppyMD.EulerDensity(model))
    fitted_estimator = benchmark(estimator.fit, data, params0=[1.0, 1.0])
    model = fitted_estimator.fetch_model()
    assert model.fitted_


@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
def test_numba_optimized(data, request, benchmark):
    n_knots = 20
    epsilon = 1e-10
    model = floppyMD.models.OverdampedFreeEnergy(np.linspace(data.stats.min - epsilon, data.stats.max + epsilon, n_knots), 1.0)
    estimator = floppyMD.LikelihoodEstimator(floppyMD.EulerNumbaOptimizedDensity(model))
    fitted_estimator = benchmark(estimator.fit, data, params0=np.concatenate((np.zeros(n_knots), np.zeros(n_knots) + 1.0)))
    model = fitted_estimator.fetch_model()
    assert model.fitted_
