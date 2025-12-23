"""
pydmc: Python implementation of DMC (Dynamic Models of Choice).

This package provides Python implementations of Dynamic Models of Choice,
specifically for the RDEX-ABCD model, using hierarchical Bayesian estimation
via PyMC (much easier to use and debug than Stan!).

Main Classes
------------
WaldStopSignalModel : WALD Stop-Signal Model for response inhibition tasks

Key Advantages of PyMC Version:
- Pure Python (easier debugging)
- Better error messages
- No compilation step
- Interactive development
- Simpler to modify and extend

Examples
--------
>>> import pandas as pd
>>> from pydmc import WaldStopSignalModel
>>>
>>> # Load your data
>>> data = pd.read_csv('stop_signal_data.csv')
>>>
>>> # Create and fit hierarchical model (PyMC!)
>>> model = WaldStopSignalModel(use_hierarchical=True)
>>> trace = model.fit(data, draws=1000, tune=1000)
>>>
>>> # Get summary
>>> model.summary()
>>>
>>> # Plot results
>>> model.plot_traces()
>>> model.plot_posterior()
"""

from .models import WaldStopSignalModel

__version__ = "0.1.0-pymc"
__author__ = "Tony Barrows"

__all__ = [
    "WaldStopSignalModel",
]
