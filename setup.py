#!/usr/bin/env python

import os
import sys
import setuptools
import versioneer

from numpy.distutils.core import setup, Extension

DISTNAME = "floppyMD"
DESCRIPTION = "Finding Langevin optimal processes for Molecular Dynamics"
AUTHOR = "Hadrien Vroylandt"
AUTHOR_EMAIL = "hadrien.vroylandt@sorbonne-universite.fr"
URL = "https://github.com/HadrienNU/"
DOWNLOAD_URL = "https://github.com/HadrienNU/"
LICENSE = "new BSD"


CLASSIFIERS = ["Intended Audience :: Science/Research", "License :: OSI Approved", "Programming Language :: Python", "Topic :: Scientific/Engineering", "Operating System :: Microsoft :: Windows", "Operating System :: POSIX", "Operating System :: Unix", "Operating System :: MacOS"]

setup(
    name=DISTNAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url=URL,
    download_url=DOWNLOAD_URL,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    packages=setuptools.find_packages(),
)
