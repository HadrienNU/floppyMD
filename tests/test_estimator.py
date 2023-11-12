import pytest
import os
import numpy as np
import floppyMD


@pytest.fixture
def data(request):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/example_2d.trj"))
    if request.param == "dask":
        trj = da.from_array(trj)
    xva_list = []
    print(trj.shape)
    for i in range(1, trj.shape[1]):
        xf = vb.xframe(trj[:, 1], trj[:, 0] - trj[0, 0])
        xvaf = vb.compute_va(xf)
        xva_list.append(xvaf)
    return xva_list


@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize("n_jobs", [1, 4])
def test_estimator(data, n_jobs, request):
    estimator = floppyMD.ExactEstimator(floppyMD.models.Overdamped())
    model = estimator.fit_fetch(data)
    assert model.shape == (198, 1, 1)
