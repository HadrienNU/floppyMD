from collections.abc import MutableSequence
import numpy as np
from ._data_statistics import traj_stats, sum_stats


class Trajectories(MutableSequence):
    def __init__(self, dt=None):
        self.dt = dt
        self.trajectories_data = []
        self.extra_data = []
        self.dim = 1
        self.stats_data = None

    def __len__(self):
        return len(self.trajectories_data)

    def __getitem__(self, i):
        return self.trajectories_data[i]

    def __delitem__(self, i):
        del self.trajectories_data[i]
        del self.extra_data[i]

    def __setitem__(self, i, v):
        self.trajectories_data[i] = (v, {})
        self.extra_data[i] = {}

    def insert(self, i, v):
        self.trajectories_data.insert(i, (v, {}))
        self.extra_data.insert(i, {})

    def __str__(self):
        return ["Trajectory of length {} and dimension {}. Extra data are {}".format(len(trj), trj[0].shape[1], list(trj[1].keys())) for trj in self.trajectories_data]

    @property
    def stats(self):
        """
        Basic statistics on the data
        """
        if self.stats_data is None:
            self.stats_data = traj_stats(self.trajectories_data[0])
            for trj in self.trajectories_data[1:]:
                self.stats_data = sum_stats(self.stats_data, traj_stats(trj))
        return self.stats_data

    @property
    def nobs(self):
        return np.sum([len(trj) for trj in self.trajectories_data])

    def to_xarray(self):
        """
        Return the data as a list of Dataset where extra variables are denoted by their name
        """
        raise NotImplementedError

    @classmethod
    def from_xarray(traj_list, traj_key="x"):
        """Take as input a list of xarray Dataset"""
        raise NotImplementedError
