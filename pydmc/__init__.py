"""
pydmc: Python implementation of DMC (Dynamic Models of Choice).

This package provides Python implementations of Dynamic Models of Choice,
specifically the RDEX-ABCD model for the ABCD study stop-signal task,
using hierarchical Bayesian estimation via PyMC.

Main Classes
------------
RDEXABCDModel : Full RDEX-ABCD model with context independence violation handling
WaldStopSignalModel : Simplified WALD stop-signal model (for basic use cases)

The RDEX-ABCD Model
-------------------
Implements the model from Weigard et al. (2023) which accounts for:
- Racing diffusion model for go process
- Ex-Gaussian distribution for stop process
- Context independence violations (v0, g parameters)
- Trigger failure and go failure
- SSD-dependent drift rates on stop trials

Key Advantages of PyMC Version:
- Pure Python (easier debugging)
- Better error messages
- No compilation step
- Interactive development
- Simpler to modify and extend

Examples
--------
>>> import pandas as pd
>>> from pydmc import RDEXABCDModel
>>>
>>> # Load your data (must have: subject, stimulus, response, rt, ssd)
>>> data = pd.read_csv('abcd_stop_signal_data.csv')
>>>
>>> # Create and fit the full RDEX-ABCD model
>>> model = RDEXABCDModel(use_hierarchical=True)
>>> trace = model.fit(data, draws=1000, tune=1000)
>>>
>>> # Get summary (includes SSRT, trigger failure, etc.)
>>> model.summary()
>>>
>>> # Plot results
>>> model.plot_traces()
>>> model.plot_posterior()
"""

from .models import WaldStopSignalModel
from .rdex_abcd import RDEXABCDModel

__version__ = "0.2.0-pymc-rdex-abcd"
__author__ = "Tony Barrows"

__all__ = [
    "RDEXABCDModel",      # Main model for ABCD data
    "WaldStopSignalModel",  # Simplified model
]
