#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
===========================
Memory Kernel Estimation
===========================

How to run memory kernel estimation
"""

import numpy as np
import matplotlib.pyplot as plt

import floppyMD

# Trouver comment on rentre les données
trj = np.loadtxt("example_2d.trj")
data = floppyMD.Trajectories(dt=trj[1, 0] - trj[0, 0])
for i in range(1, trj.shape[1]):
    data.append(trj[:, 1:2])

bf = floppyMD.function_basis.BSplines(5).fit(data)
model = floppyMD.models.OverdampedBF(bf)
estimator = floppyMD.KramersMoyalEstimator(model)
model = estimator.fit_fetch(data)

# To find a correct parametrization of the space
bins = np.histogram_bin_edges(data[0]["x"], bins=15)
xfa = (bins[1:] + bins[:-1]) / 2.0


fig, axs = plt.subplots(1, 2)
# Force plot
axs[0].set_title("Force")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$F(x)$")
axs[0].grid()
axs[0].plot(xfa, model.force(xfa.reshape(-1, 1)))

# Diffusion plot
axs[1].set_title("Diffusion")
axs[1].grid()
axs[1].plot(xfa, model.diffusion(xfa.reshape(-1, 1)))
axs[1].set_xlabel("$x$")
axs[1].set_ylabel("$D(x)$")
plt.show()
