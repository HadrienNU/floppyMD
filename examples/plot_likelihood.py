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
data.append(trj[:, 1:2])

model = floppyMD.models.BrownianMotion()

fig, axs = plt.subplots(1, 2)
axs[0].set_title("Force")
axs[0].set_xlabel("$f$")
axs[0].set_ylabel("$L(f,1.0)$")
axs[0].grid()
axs[0].legend()


axs[1].set_title("Diffusion")
axs[1].grid()
axs[1].set_xlabel("$\\sigma$")
axs[1].set_ylabel("$L(1.0,\\sigma)$")
axs[1].legend()


force_range = np.linspace(-1, 1, 25)
diff_range = np.linspace(0.5, 2, 25)


for name, transitioncls in zip(
    ["Euler", "Ozaki", "ShojiOzaki", "Elerian", "Kessler", "Drozdov"],
    [
        floppyMD.EulerDensity,
        floppyMD.OzakiDensity,
        floppyMD.ShojiOzakiDensity,
        floppyMD.ElerianDensity,
        floppyMD.KesslerDensity,
        floppyMD.DrozdovDensity,
    ],
):
    likelihood = transitioncls(model)
    likelihood_vals_force = np.zeros_like(force_range)
    for n, f in enumerate(force_range):
        likelihood_vals_force[n] = likelihood(1.0, data[0], np.array([f, 1.0]))[0]
    axs[0].plot(force_range, likelihood_vals_force, label=name)
    likelihood_vals_diff = np.zeros_like(diff_range)
    for n, d in enumerate(diff_range):
        likelihood_vals_diff[n] = likelihood(1.0, data[0], np.array([1.0, d]))[0]

    axs[1].plot(diff_range, likelihood_vals_diff, label=name)


plt.show()
