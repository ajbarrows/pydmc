"""
pydmc: Python implementation of DMC (Dynamic Models of Choice).

This package provides Python implementations of Dynamic Models of Choice,
specifically for the RDEX-ABCD model, using hierarchical Bayesian estimation
via Stan.

Main Classes
------------
WaldStopSignalModel : WALD Stop-Signal Model for response inhibition tasks
StanBackend : Backend handler for Stan sampling interfaces

Examples
--------
>>> import pandas as pd
>>> from pydmc import WaldStopSignalModel
>>>
>>> # Load your data
>>> data = pd.read_csv('stop_signal_data.csv')
>>>
>>> # Create and fit hierarchical model
>>> model = WaldStopSignalModel(use_hierarchical=True)
>>> fit = model.fit(data, chains=4, iter=2000)
>>>
>>> # Get parameter estimates
>>> estimates = model.get_parameter_estimates()
>>> print(estimates)
"""

from .backends import StanBackend
from .models import WaldStopSignalModel
from .utils import setup_hpc_environment, check_environment, print_environment_info

__version__ = "0.0.1"
__author__ = "Tony Barrows"

__all__ = [
    "WaldStopSignalModel",
    "StanBackend",
    "setup_hpc_environment",
    "check_environment",
    "print_environment_info",
]
