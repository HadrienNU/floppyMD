from collections import namedtuple
import numpy as np

DescribeResult = namedtuple("DescribeResult", ("nobs", "minmax", "mean", "variance"))


def quick_describe(X):
    """
    Simply return the dimension of the data
    """
    nobs, dim = X.shape
    return DescribeResult(nobs, (np.zeros(dim), np.zeros(dim)), np.zeros(dim), np.zeros(dim))


def minimal_describe(X):
    """
    Simply return the dimension of the data
    """
    nobs, dim = X.shape
    return DescribeResult(nobs, (np.min(X, axis=0), np.max(X, axis=0)), np.mean(X, axis=0), np.zeros(dim))


def sum_describe(d1, d2):
    return DescribeResult(d1.nobs + d2.nobs, (np.minimum(d1.minmax[0], d2.minmax[0]), np.maximum(d1.minmax[1], d2.minmax[1])), (d1.mean * d1.nobs + d2.mean * d2.nobs) / (d1.nobs + d2.nobs), d1.variance)


# Il faudrait faire un calcul de descripteurs avec accumulation sur les trajs ?


# TODO: Add some function that allow to take a list of keywords arguments and make a describe from it, so we can deal with various type of input into the fit function
