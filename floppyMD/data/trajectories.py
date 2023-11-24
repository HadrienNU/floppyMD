import collections
import numpy as np
from ._data_statistics import DescribeResult, sum_describe


class Trajectories(collections.MutableSequence):
    def __init__(self, dt=None):
        self.dt = dt
        self.trajectories_data = []
        self.dim = 1
        self.describe_data = None

    def __len__(self):
        return len(self.trajectories_data)

    def __getitem__(self, i):
        return self.trajectories_data[i]

    def __delitem__(self, i):
        del self.trajectories_data[i]

    def __setitem__(self, i, v):
        self.trajectories_data[i] = v

    def insert(self, i, v):
        self.trajectories_data.insert(i, v)

    def __str__(self):
        return ["Trajectory of length {} and dimension {}".format(len(trj), trj.shape[1]) for trj in self.trajectories_data]

    @property
    def stats(self):
        """
        Basic statistics on the data
        """
        if self.describe_data is None:
            self.describe_data = DescribeResult(self.nobs, (np.zeros(self.dim), np.zeros(self.dim)), np.mean([np.mean(trj, axis=0) for trj in self.trajectories_data]), np.zeros(self.dim))
        return self.describe_data

    @property
    def nobs(self):
        return np.sum([len(trj) for trj in self.trajectories_data])
