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
xva_list = []
print(trj.shape)
for i in range(1, trj.shape[1]):
    xf = vb.xframe(trj[:, i], trj[:, 0] - trj[0, 0])
    xvaf = vb.compute_va(xf)
    xva_list.append(xvaf)

model = floppyMD.models.Overdamped()
estimator = floppyMD.ExactEstimator(model)
model = estimator.fit_fetch(data)

# To find a correct parametrization of the space
bins = np.histogram_bin_edges(xvaf["x"], bins=15)
xfa = (bins[1:] + bins[:-1]) / 2.0


fig_kernel, axs = plt.subplots(1, 2)
# Force plot
axs[0].set_title("Force")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$-dU(x)/dx$")
axs[0].grid()
axs[0].plot(xfa, model.force(xfa))
# Diffusion plot
axs[1].set_title("Diffusion")
axs[1].grid()
axs[1].plot(xfa, model.diffusion(xfa))
# axs[1].plot(time, kernel[:, :, 0, 0], "-x")
axs[1].set_xlabel("$t$")
axs[1].set_ylabel("$D$")
plt.show()
