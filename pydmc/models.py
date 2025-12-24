"""
WALD Stop-Signal Model Implementation using PyMC.

This is a PyMC-based implementation that's easier to debug and modify
than the Stan version. It uses pure Python and provides better error messages.

Key advantages over Stan:
- Pure Python (easier debugging)
- Better error messages
- Interactive development
- No compilation step
- Easier to modify and extend
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from typing import Dict, Optional, Any
import json


class WaldStopSignalModel:
    """
    WALD Stop-Signal Model for response inhibition tasks using PyMC.

    This implementation is much simpler and easier to debug than Stan.

    Parameters
    ----------
    use_hierarchical : bool, default=True
        Whether to use hierarchical modeling (multiple subjects)

    Examples
    --------
    >>> import pandas as pd
    >>> from pydmc import WaldStopSignalModel
    >>>
    >>> # Load data
    >>> data = pd.read_csv('stop_signal_data.csv')
    >>>
    >>> # Create model and check priors
    >>> model = WaldStopSignalModel(use_hierarchical=True)
    >>> model.prepare_data(data)
    >>> model.build_model()
    >>> model.prior_predictive_check()  # Validate priors before fitting
    >>>
    >>> # Fit model
    >>> trace = model.fit(data, draws=1000, tune=1000)
    >>>
    >>> # Get results and diagnostics
    >>> model.summary()
    >>> model.plot_traces()
    >>> model.posterior_predictive_check()  # Validate model fit
    """

    def __init__(self, use_hierarchical: bool = True):
        """Initialize the WALD Stop-Signal model."""
        self.use_hierarchical = use_hierarchical
        self.model = None
        self.trace = None
        self.data = None

    def prepare_data(self, data_df: pd.DataFrame) -> Dict:
        """
        Prepare data for PyMC model.

        Parameters
        ----------
        data_df : pd.DataFrame
            DataFrame with columns: subject, stimulus, response, rt, ssd (optional)

        Returns
        -------
        dict
            Prepared data dictionary
        """
        # Validate required columns
        required_cols = ['subject', 'stimulus', 'response', 'rt']
        missing_cols = [col for col in required_cols if col not in data_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Create working copy
        df = data_df.copy()

        # Filter valid trials
        df = df[df['rt'] > 0.15].copy()  # Need RT > non-decision time
        df = df[df['response'].isin([0, 1, 2])].copy()

        if len(df) == 0:
            raise ValueError("No valid trials after filtering")

        # Create stop trial indicator
        if 'ssd' in df.columns:
            df['is_stop'] = (~df['ssd'].isna()).astype(int)
        else:
            df['is_stop'] = 0

        # Create correct indicator
        df['correct'] = (
            ((df['stimulus'] == 0) & (df['response'] == 1)) |
            ((df['stimulus'] == 1) & (df['response'] == 2))
        ).astype(int)

        # Map subjects to integers
        if self.use_hierarchical:
            unique_subjects = sorted(df['subject'].unique())
            subject_map = {subj: i for i, subj in enumerate(unique_subjects)}
            df['subject_idx'] = df['subject'].map(subject_map)
            n_subjects = len(unique_subjects)
        else:
            df['subject_idx'] = 0
            n_subjects = 1

        # Only use go trials with responses for now (simplified)
        df_go = df[(df['is_stop'] == 0) & (df['response'] > 0)].copy()

        prepared_data = {
            'n_obs': len(df_go),
            'n_subjects': n_subjects,
            'subject_idx': df_go['subject_idx'].values,
            'rt': df_go['rt'].values,
            'correct': df_go['correct'].values,
            'response': df_go['response'].values,
            'stimulus': df_go['stimulus'].values,
        }

        self.data = prepared_data
        return prepared_data

    def build_model(self) -> pm.Model:
        """
        Build the PyMC model.

        Returns
        -------
        pm.Model
            PyMC model object
        """
        if self.data is None:
            raise ValueError("Must prepare data first with prepare_data()")

        with pm.Model() as model:
            if self.use_hierarchical:
                # Hierarchical parameters (much simpler than Stan!)
                # Group-level means
                mu_B = pm.Lognormal('mu_B', mu=0, sigma=1)
                mu_t0 = pm.Beta('mu_t0', alpha=3, beta=17)  # ~0.15
                mu_v_true = pm.Normal('mu_v_true', mu=2, sigma=1)
                mu_v_false = pm.Normal('mu_v_false', mu=1, sigma=1)

                # Group-level standard deviations
                sigma_B = pm.HalfNormal('sigma_B', sigma=0.5)
                sigma_t0 = pm.HalfNormal('sigma_t0', sigma=0.1)
                sigma_v_true = pm.HalfNormal('sigma_v_true', sigma=0.5)
                sigma_v_false = pm.HalfNormal('sigma_v_false', sigma=0.5)

                # Individual parameters (centered parameterization)
                B = pm.Lognormal('B', mu=pm.math.log(mu_B), sigma=sigma_B,
                                shape=self.data['n_subjects'])
                t0 = pm.TruncatedNormal('t0', mu=mu_t0, sigma=sigma_t0,
                                       lower=0, upper=0.5,
                                       shape=self.data['n_subjects'])
                v_true = pm.Normal('v_true', mu=mu_v_true, sigma=sigma_v_true,
                                  shape=self.data['n_subjects'])
                v_false = pm.Normal('v_false', mu=mu_v_false, sigma=sigma_v_false,
                                   shape=self.data['n_subjects'])

                # Get parameters for each trial
                B_trial = B[self.data['subject_idx']]
                t0_trial = t0[self.data['subject_idx']]
                v_true_trial = v_true[self.data['subject_idx']]
                v_false_trial = v_false[self.data['subject_idx']]

            else:
                # Individual-level model (even simpler!)
                B_trial = pm.Lognormal('B', mu=0, sigma=1)
                t0_trial = pm.Beta('t0', alpha=3, beta=17)
                v_true_trial = pm.Normal('v_true', mu=2, sigma=1)
                v_false_trial = pm.Normal('v_false', mu=1, sigma=1)

            # Drift rate based on correctness
            v_trial = pm.math.switch(
                self.data['correct'],
                v_true_trial,
                v_false_trial
            )

            # Wald (Inverse Gaussian) likelihood
            # mu = boundary / drift_rate
            # lambda = boundary^2
            mu_wald = B_trial / v_trial
            lambda_wald = B_trial ** 2

            # Decision time = RT - non-decision time
            dt = self.data['rt'] - t0_trial

            # Wald log-likelihood (using pm.Potential for custom likelihood)
            # The Wald distribution log-pdf is:
            # log_wald(x; mu, lambda) = 0.5*log(lambda/(2*pi*x^3)) - lambda*(x-mu)^2/(2*mu^2*x)
            log_wald = (
                0.5 * pm.math.log(lambda_wald / (2 * np.pi * dt**3))
                - lambda_wald * (dt - mu_wald)**2 / (2 * mu_wald**2 * dt)
            )
            pm.Potential('obs', log_wald)

        self.model = model
        return model

    def fit(self, data_df: pd.DataFrame, draws: int = 1000, tune: int = 1000,
            chains: int = 4, cores: Optional[int] = None,
            **kwargs) -> az.InferenceData:
        """
        Fit the model using PyMC's NUTS sampler.

        Parameters
        ----------
        data_df : pd.DataFrame
            Data with required columns
        draws : int, default=1000
            Number of samples to draw
        tune : int, default=1000
            Number of tuning steps
        chains : int, default=4
            Number of MCMC chains
        cores : int, optional
            Number of cores for parallel sampling
        **kwargs
            Additional arguments for pm.sample()

        Returns
        -------
        arviz.InferenceData
            Posterior samples
        """
        # Prepare data
        print("Preparing data...")
        self.prepare_data(data_df)

        # Build model
        print("Building PyMC model...")
        self.build_model()

        # Sample
        print(f"Sampling {chains} chains with {draws} draws each...")
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores if cores else chains,
                **kwargs
            )

        print("✓ Sampling complete!")
        return self.trace

    def summary(self) -> pd.DataFrame:
        """
        Print and return model summary.

        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        if self.trace is None:
            print("Model has not been fitted yet.")
            return None

        summary_df = az.summary(self.trace)
        print(summary_df)
        return summary_df

    def plot_traces(self, var_names: Optional[list] = None, **kwargs):
        """
        Plot MCMC traces.

        Parameters
        ----------
        var_names : list, optional
            Variables to plot. If None, plots all.
        **kwargs
            Additional arguments for az.plot_trace()
        """
        if self.trace is None:
            print("Model has not been fitted yet.")
            return

        if var_names is None:
            if self.use_hierarchical:
                var_names = ['mu_B', 'mu_t0', 'mu_v_true', 'mu_v_false',
                           'sigma_B', 'sigma_t0', 'sigma_v_true', 'sigma_v_false']
            else:
                var_names = ['B', 't0', 'v_true', 'v_false']

        az.plot_trace(self.trace, var_names=var_names, **kwargs)
        plt.tight_layout()
        plt.show()

    def plot_posterior(self, var_names: Optional[list] = None, **kwargs):
        """
        Plot posterior distributions.

        Parameters
        ----------
        var_names : list, optional
            Variables to plot
        **kwargs
            Additional arguments for az.plot_posterior()
        """
        if self.trace is None:
            print("Model has not been fitted yet.")
            return

        az.plot_posterior(self.trace, var_names=var_names, **kwargs)
        plt.tight_layout()
        plt.show()

    def save_results(self, filepath: str):
        """
        Save results to NetCDF format.

        Parameters
        ----------
        filepath : str
            Output file path (.nc)
        """
        if self.trace is None:
            raise ValueError("Model has not been fitted yet.")

        self.trace.to_netcdf(filepath)
        print(f"Results saved to: {filepath}")

    def load_results(self, filepath: str) -> az.InferenceData:
        """
        Load results from NetCDF format.

        Parameters
        ----------
        filepath : str
            Input file path (.nc)

        Returns
        -------
        arviz.InferenceData
            Loaded posterior samples
        """
        self.trace = az.from_netcdf(filepath)
        print(f"Results loaded from: {filepath}")
        return self.trace

    def prior_predictive_check(self, samples: int = 500, **kwargs):
        """
        Perform prior predictive checks to validate prior choices.

        This samples from the prior distributions and generates predictions
        to check if the priors produce reasonable values before seeing data.

        Parameters
        ----------
        samples : int, default=500
            Number of prior samples to draw
        **kwargs
            Additional arguments for pm.sample_prior_predictive()

        Returns
        -------
        arviz.InferenceData
            Prior predictive samples
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call prepare_data() and build_model() first.")

        print(f"Sampling {samples} prior predictive samples...")
        with self.model:
            prior_pred = pm.sample_prior_predictive(samples=samples, **kwargs)

        # Plot prior predictive distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot prior distributions for key parameters
        if self.use_hierarchical:
            param_names = ['mu_B', 'mu_t0', 'mu_v_true', 'mu_v_false']
            titles = ['Prior: Boundary (B)', 'Prior: Non-decision time (t0)',
                     'Prior: Drift rate (correct)', 'Prior: Drift rate (error)']
        else:
            param_names = ['B', 't0', 'v_true', 'v_false']
            titles = ['Prior: Boundary (B)', 'Prior: Non-decision time (t0)',
                     'Prior: Drift rate (correct)', 'Prior: Drift rate (error)']

        for ax, param, title in zip(axes.flat, param_names, titles):
            if param in prior_pred.prior:
                samples_flat = prior_pred.prior[param].values.flatten()
                ax.hist(samples_flat, bins=50, alpha=0.7, edgecolor='black')
                ax.set_title(title)
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.axvline(samples_flat.mean(), color='red', linestyle='--',
                          label=f'Mean: {samples_flat.mean():.3f}')
                ax.legend()

        plt.tight_layout()
        plt.show()

        # Also plot the prior predictive for observed data if available
        if 'obs' in prior_pred.prior_predictive:
            fig, ax = plt.subplots(figsize=(10, 6))
            obs_samples = prior_pred.prior_predictive['obs'].values.flatten()
            # Only plot reasonable RT values
            obs_samples = obs_samples[(obs_samples > 0) & (obs_samples < 5)]
            ax.hist(obs_samples, bins=100, alpha=0.7, edgecolor='black')
            ax.set_title('Prior Predictive: Decision Time Distribution')
            ax.set_xlabel('Decision Time (s)')
            ax.set_ylabel('Frequency')
            ax.axvline(obs_samples.mean(), color='red', linestyle='--',
                      label=f'Mean: {obs_samples.mean():.3f}s')
            if self.data is not None:
                actual_dt = self.data['rt'] - 0.15  # Approximate
                ax.axvline(actual_dt.mean(), color='green', linestyle='--',
                          label=f'Actual mean: {actual_dt.mean():.3f}s', linewidth=2)
            ax.legend()
            plt.tight_layout()
            plt.show()

        print("✓ Prior predictive check complete!")
        return prior_pred

    def posterior_predictive_check(self, samples: Optional[int] = None, **kwargs):
        """
        Perform posterior predictive checks.

        This generates predictions from the fitted posterior to check
        if the model can reproduce the observed data patterns.

        Parameters
        ----------
        samples : int, optional
            Number of posterior predictive samples. If None, uses all posterior samples.
        **kwargs
            Additional arguments for pm.sample_posterior_predictive()

        Returns
        -------
        arviz.InferenceData
            Posterior predictive samples
        """
        if self.trace is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        print("Generating posterior predictive samples...")
        with self.model:
            if samples is not None:
                kwargs['predictions'] = samples
            ppc = pm.sample_posterior_predictive(self.trace, **kwargs)

        # Plot posterior predictive check
        az.plot_ppc(ppc, num_pp_samples=100)
        plt.tight_layout()
        plt.show()

        print("✓ Posterior predictive check complete!")
        return ppc
