"""
Stan backend handling for pydmc.

This module provides a unified interface for different Stan backends
(CmdStanPy and PyStan), allowing flexible MCMC sampling for DMC models.
"""

import os
import tempfile
from typing import Any, Dict, Optional
import warnings


class StanBackend:
    """
    Handles Stan backend selection and interface.

    This class automatically detects and uses the best available Stan backend,
    preferring CmdStanPy (recommended) over PyStan. It provides a unified
    interface for compiling models and sampling from them.

    Attributes
    ----------
    backend : module
        The Stan backend module (cmdstanpy or pystan)
    backend_name : str
        Name of the backend being used ('cmdstanpy' or 'pystan')

    Raises
    ------
    ImportError
        If no Stan backend is available
    """

    def __init__(self):
        """Initialize the best available Stan backend."""
        self.backend = None
        self.backend_name = None
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the best available Stan backend."""

        # Try CmdStanPy first (recommended)
        try:
            import cmdstanpy as stan
            self.backend = stan
            self.backend_name = "cmdstanpy"
            return
        except ImportError:
            pass

        # Try PyStan as fallback
        try:
            import pystan
            self.backend = pystan
            self.backend_name = "pystan"
            return
        except ImportError:
            pass

        # No Stan backend available
        raise ImportError(
            "No Stan backend available. Please install one of:\n"
            "  pip install cmdstanpy  # Recommended\n"
            "  pip install pystan     # Alternative"
        )

    def compile_model(self, model_code: str, data: Optional[Dict] = None) -> Any:
        """
        Compile Stan model using available backend.

        Parameters
        ----------
        model_code : str
            Stan model code as a string
        data : dict, optional
            Data dictionary (unused, kept for API compatibility)

        Returns
        -------
        model : StanModel
            Compiled Stan model object

        Raises
        ------
        RuntimeError
            If no valid Stan backend is initialized
        """

        if self.backend_name == "cmdstanpy":
            # Write model to temporary file for CmdStanPy
            with tempfile.NamedTemporaryFile(mode='w', suffix='.stan', delete=False) as f:
                f.write(model_code)
                model_file = f.name

            try:
                model = self.backend.CmdStanModel(stan_file=model_file)
                return model
            finally:
                # Clean up temporary file
                if os.path.exists(model_file):
                    os.unlink(model_file)

        elif self.backend_name == "pystan":
            return self.backend.StanModel(model_code=model_code)

        else:
            raise RuntimeError("No valid Stan backend initialized")

    def sample(self, model: Any, data: Dict, **kwargs) -> Any:
        """
        Sample from compiled model using available backend.

        Parameters
        ----------
        model : StanModel
            Compiled Stan model
        data : dict
            Data dictionary for Stan
        **kwargs : dict
            Sampling parameters:
            - chains : int, number of MCMC chains (default: 4)
            - iter : int, total iterations per chain (default: 1000)
            - warmup : int, warmup iterations (default: 500)
            - cores : int, number of parallel cores (default: 4)
            - show_progress : bool, show sampling progress (default: True)

        Returns
        -------
        fit : StanFit
            Fitted model object containing posterior samples

        Raises
        ------
        RuntimeError
            If no valid Stan backend is initialized
        """

        if self.backend_name == "cmdstanpy":
            # CmdStanPy interface
            return model.sample(
                data=data,
                chains=kwargs.get('chains', 4),
                iter_sampling=kwargs.get('iter', 1000) - kwargs.get('warmup', 500),
                iter_warmup=kwargs.get('warmup', 500),
                parallel_chains=kwargs.get('cores', 4),
                show_progress=kwargs.get('show_progress', True)
            )

        elif self.backend_name == "pystan":
            # PyStan interface
            return model.sampling(
                data=data,
                chains=kwargs.get('chains', 4),
                iter=kwargs.get('iter', 1000),
                warmup=kwargs.get('warmup', 500),
                n_jobs=kwargs.get('cores', 4),
                verbose=kwargs.get('show_progress', True)
            )

        else:
            raise RuntimeError("No valid Stan backend initialized")

    def extract_samples(self, fit: Any) -> Dict[str, Any]:
        """
        Extract samples from fit object.

        Parameters
        ----------
        fit : StanFit
            Fitted model object

        Returns
        -------
        samples : dict
            Dictionary of parameter names to posterior samples

        Raises
        ------
        RuntimeError
            If no valid Stan backend is initialized
        """

        if self.backend_name == "cmdstanpy":
            return fit.stan_variables()

        elif self.backend_name == "pystan":
            return fit.extract()

        else:
            raise RuntimeError("No valid Stan backend initialized")
