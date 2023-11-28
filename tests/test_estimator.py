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
    print(trj.shape)
    trj_list = floppyMD.Trajectories(dt=trj[1, 0] - trj[0, 0])
    for i in range(1, trj.shape[1]):
        trj_list.append(trj[:, 1:2])
    return trj_list


@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
def test_direct_estimator(data, request):
    bf = floppyMD.function_basis.Linear().fit(data)
    model = floppyMD.models.OverdampedBF(bf)
    estimator = floppyMD.KramersMoyalEstimator(model)
    model = estimator.fit_fetch(data)
    assert model.fitted_


@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
def test_likelihood_estimator(data, request):
    bf = floppyMD.function_basis.Linear().fit(data)
    model = floppyMD.models.OverdampedBF(bf)
    estimator = floppyMD.LikelihoodEstimator(floppyMD.EulerDensity(model))
    model = estimator.fit_fetch(data, params0=[1.0, 1.0])
    assert model.fitted_
