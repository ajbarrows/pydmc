"""
Stop-Signal Model Implementations using PyMC.

This module contains two stop-signal models:

1. WaldStopSignalModel - Simplified baseline model for comparison
   - Basic Wald (Inverse Gaussian) model for go trials
   - Easier to understand and debug
   - Does NOT account for context independence violations

2. RDEXABCDModel - Full RDEX-ABCD model for ABCD study data
   - Racing diffusion model for go process
   - Ex-Gaussian distribution for stop process
   - Accounts for context independence violations with SSD-dependent drift rates
   - Handles trigger failure and go failure parameters

Both are PyMC-based implementations that are easier to debug and modify
than Stan versions.

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
from scipy import stats


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


# =============================================================================
# RDEX-ABCD Model - Full production model for ABCD study
# =============================================================================


class RDEXABCDModel:
    """
    RDEX-ABCD Stop-Signal Model for the ABCD study using PyMC.

    This model accounts for context independence violations in the ABCD task
    where the visual stop signal replaces the go stimulus, limiting the time
    available for stimulus processing on stop trials.

    Parameters
    ----------
    use_hierarchical : bool, default=True
        Whether to use hierarchical modeling (multiple subjects)

    Model Parameters
    ----------------
    Go Process (Racing Diffusion):
        - t0: Non-decision time
        - B: Evidence threshold
        - v_plus: Matching accumulator drift rate (asymptotic)
        - v_minus: Mismatching accumulator drift rate (asymptotic)
        - v0: Processing speed (base rate with no discrimination)
        - g: Perceptual growth rate (how fast discrimination info grows with SSD)
        - pgf: Probability of go failure

    Stop Process (Ex-Gaussian):
        - mu: Ex-Gaussian normal mean
        - sigma: Ex-Gaussian normal SD
        - tau: Ex-Gaussian exponential mean
        - ptf: Probability of trigger failure

    Examples
    --------
    >>> model = RDEXABCDModel(use_hierarchical=True)
    >>> model.prepare_data(data)
    >>> model.build_model()
    >>> trace = model.fit(data, draws=1000, tune=1000)
    >>> model.summary()
    """

    def __init__(self, use_hierarchical: bool = True):
        """Initialize the RDEX-ABCD model."""
        self.use_hierarchical = use_hierarchical
        self.model = None
        self.trace = None
        self.data = None

    def prepare_data(self, data_df: pd.DataFrame) -> Dict:
        """
        Prepare data for RDEX-ABCD model.

        Parameters
        ----------
        data_df : pd.DataFrame
            DataFrame with columns:
            - subject: Subject identifier
            - stimulus: Stimulus type (0=left, 1=right or similar)
            - response: Response (0=no response, 1=left, 2=right or similar)
            - rt: Response time in seconds
            - ssd: Stop-signal delay in seconds (NaN for go trials)

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

        # Identify stop vs go trials
        if 'ssd' in df.columns:
            df['is_stop'] = (~df['ssd'].isna()).astype(int)
            df['ssd'] = df['ssd'].fillna(0)  # Fill NaN with 0 for go trials
        else:
            df['is_stop'] = 0
            df['ssd'] = 0

        # Create match indicator (does response match stimulus?)
        # Assuming stimulus 0 -> response 1 is a match, stimulus 1 -> response 2 is a match
        df['match'] = (
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

        # Separate go and stop trials
        df_go = df[(df['is_stop'] == 0) & (df['response'] > 0)].copy()
        df_stop = df[df['is_stop'] == 1].copy()

        # Further separate stop trials into signal-respond and successful stops
        df_signal_respond = df_stop[df_stop['response'] > 0].copy()
        df_successful_stop = df_stop[df_stop['response'] == 0].copy()

        # Filter out very fast/slow RTs (likely contaminants)
        df_go = df_go[(df_go['rt'] >= 0.15) & (df_go['rt'] <= 2.0)].copy()
        df_signal_respond = df_signal_respond[
            (df_signal_respond['rt'] >= 0.15) & (df_signal_respond['rt'] <= 2.0)
        ].copy()

        prepared_data = {
            # Go trials
            'n_go': len(df_go),
            'go_subject_idx': df_go['subject_idx'].values,
            'go_rt': df_go['rt'].values,
            'go_match': df_go['match'].values,
            'go_stimulus': df_go['stimulus'].values,

            # Signal-respond trials
            'n_signal_respond': len(df_signal_respond),
            'sr_subject_idx': df_signal_respond['subject_idx'].values,
            'sr_rt': df_signal_respond['rt'].values,
            'sr_match': df_signal_respond['match'].values,
            'sr_ssd': df_signal_respond['ssd'].values,
            'sr_stimulus': df_signal_respond['stimulus'].values,

            # Successful stop trials
            'n_successful_stop': len(df_successful_stop),
            'ss_subject_idx': df_successful_stop['subject_idx'].values,
            'ss_ssd': df_successful_stop['ssd'].values,
            'ss_stimulus': df_successful_stop['stimulus'].values,

            # Subject info
            'n_subjects': n_subjects,
        }

        self.data = prepared_data
        print(f"Data prepared: {prepared_data['n_go']} go trials, "
              f"{prepared_data['n_signal_respond']} signal-respond trials, "
              f"{prepared_data['n_successful_stop']} successful stops, "
              f"{prepared_data['n_subjects']} subjects")
        return prepared_data

    def _compute_drift_rates_for_ssd(self, v_plus, v_minus, v0, g, ssd):
        """
        Compute drift rates for matching and mismatching accumulators at a given SSD.

        At SSD=0, both rates equal v0 (no discriminative information).
        As SSD increases, discrimination grows linearly at rate g until
        reaching asymptotic values v_plus and v_minus.

        Parameters
        ----------
        v_plus : tensor
            Asymptotic matching accumulator rate
        v_minus : tensor
            Asymptotic mismatching accumulator rate
        v0 : tensor
            Processing speed (base rate)
        g : tensor
            Perceptual growth rate
        ssd : tensor
            Stop-signal delay(s)

        Returns
        -------
        tuple
            (matching_rate, mismatching_rate)
        """
        # Discrimination components grow from 0 at SSD=0
        # matching discrimination: (v_plus - v0)
        # mismatching discrimination: (v_minus - v0)

        # Linear growth: discrimination = g * SSD, capped at asymptotic values
        # matching_rate = v0 + min(g * SSD, v_plus - v0)
        # mismatching_rate = v0 + min(g * SSD, v_minus - v0)

        matching_discrimination = pm.math.minimum(g * ssd, v_plus - v0)
        mismatching_discrimination = pm.math.minimum(g * ssd, v_minus - v0)

        matching_rate = v0 + matching_discrimination
        mismatching_rate = v0 + mismatching_discrimination

        return matching_rate, mismatching_rate

    def _wald_logp(self, rt, mu, lam):
        """
        Compute log-probability for Wald (Inverse Gaussian) distribution.

        Parameters
        ----------
        rt : tensor
            Response times (decision times)
        mu : tensor
            Mean parameter (B / drift_rate)
        lam : tensor
            Shape parameter (B^2)

        Returns
        -------
        tensor
            Log-probabilities
        """
        # Wald log-pdf: 0.5*log(lam/(2*pi*rt^3)) - lam*(rt-mu)^2/(2*mu^2*rt)
        return (
            0.5 * pm.math.log(lam / (2 * np.pi * rt**3))
            - lam * (rt - mu)**2 / (2 * mu**2 * rt)
        )

    def _exgaussian_cdf(self, x, mu, sigma, tau):
        """
        Compute CDF of ex-Gaussian distribution.

        Uses numerical approximation suitable for PyMC tensors.
        """
        # For numerical stability, use the relationship:
        # ExGaussian CDF ≈ Φ((x-mu)/sigma) for computational efficiency
        # More accurate version would integrate, but this is a reasonable approximation
        # that works with PyMC's automatic differentiation

        # Add small epsilon to sigma and tau to prevent division by zero
        sigma_safe = pm.math.maximum(sigma, 1e-6)
        tau_safe = pm.math.maximum(tau, 1e-6)

        # Standardize
        z = (x - mu - tau_safe * (tau_safe / sigma_safe)**2) / sigma_safe
        # Clip z to prevent extreme values in erf
        z = pm.math.clip(z, -10, 10)
        # Normal CDF component
        normal_cdf = 0.5 * (1 + pm.math.erf(z / pm.math.sqrt(2)))

        # Correction factor for exponential component (with clipping to prevent overflow)
        exponent = tau_safe / sigma_safe**2 * (x - mu) + 0.5 * (tau_safe / sigma_safe)**2
        exponent = pm.math.clip(exponent, -20, 20)  # Prevent exp overflow

        z_correction = (x - mu - tau_safe**2/sigma_safe**2) / (sigma_safe * pm.math.sqrt(2))
        z_correction = pm.math.clip(z_correction, -10, 10)

        correction = pm.math.exp(exponent) * (1 - 0.5 * (1 + pm.math.erf(z_correction)))

        return normal_cdf - correction

    def build_model(self) -> pm.Model:
        """
        Build the RDEX-ABCD PyMC model.

        Returns
        -------
        pm.Model
            PyMC model object
        """
        if self.data is None:
            raise ValueError("Must prepare data first with prepare_data()")

        with pm.Model() as model:
            if self.use_hierarchical:
                # Group-level means (hyperparameters)
                mu_B = pm.Lognormal('mu_B', mu=-0.2, sigma=0.3)  # ~0.7-1.0
                mu_t0 = pm.Beta('mu_t0', alpha=3, beta=17)  # ~0.15
                # CRITICAL FIX: Use TruncatedNormal with lower bounds to prevent negative drift rates
                mu_v_plus = pm.TruncatedNormal('mu_v_plus', mu=3, sigma=0.5, lower=1.0)
                mu_v_minus = pm.TruncatedNormal('mu_v_minus', mu=0.8, sigma=0.3, lower=0.3)
                mu_v0 = pm.TruncatedNormal('mu_v0', mu=2.2, sigma=0.4, lower=1.5)
                mu_g = pm.Gamma('mu_g', alpha=4, beta=1.5)  # ~2-3

                mu_stop_mu = pm.TruncatedNormal('mu_stop_mu', mu=0.25, sigma=0.1, lower=0, upper=0.5)
                mu_stop_sigma = pm.HalfNormal('mu_stop_sigma', sigma=0.1)
                mu_stop_tau = pm.HalfNormal('mu_stop_tau', sigma=0.1)

                # Probit-scale for failure probabilities
                mu_pgf_probit = pm.Normal('mu_pgf_probit', mu=-2, sigma=1)
                mu_ptf_probit = pm.Normal('mu_ptf_probit', mu=-2, sigma=1)

                # Group-level standard deviations
                sigma_B = pm.HalfNormal('sigma_B', sigma=0.5)
                sigma_t0 = pm.HalfNormal('sigma_t0', sigma=0.05)
                sigma_v_plus = pm.HalfNormal('sigma_v_plus', sigma=0.5)
                sigma_v_minus = pm.HalfNormal('sigma_v_minus', sigma=0.5)
                sigma_v0 = pm.HalfNormal('sigma_v0', sigma=0.5)
                sigma_g = pm.HalfNormal('sigma_g', sigma=1)

                sigma_stop_mu = pm.HalfNormal('sigma_stop_mu', sigma=0.05)
                sigma_stop_sigma = pm.HalfNormal('sigma_stop_sigma', sigma=0.05)
                sigma_stop_tau = pm.HalfNormal('sigma_stop_tau', sigma=0.05)

                sigma_pgf_probit = pm.HalfNormal('sigma_pgf_probit', sigma=0.5)
                sigma_ptf_probit = pm.HalfNormal('sigma_ptf_probit', sigma=0.5)

                # Individual-level parameters
                B = pm.Lognormal('B', mu=pm.math.log(mu_B), sigma=sigma_B,
                                shape=self.data['n_subjects'])
                t0 = pm.TruncatedNormal('t0', mu=mu_t0, sigma=sigma_t0,
                                       lower=0.05, upper=0.5,
                                       shape=self.data['n_subjects'])
                # CRITICAL FIX: Add lower bounds to prevent negative drift rates
                v_plus = pm.TruncatedNormal('v_plus', mu=mu_v_plus, sigma=sigma_v_plus,
                                           lower=1.0,
                                           shape=self.data['n_subjects'])
                v_minus = pm.TruncatedNormal('v_minus', mu=mu_v_minus, sigma=sigma_v_minus,
                                            lower=0.3,
                                            shape=self.data['n_subjects'])
                v0 = pm.TruncatedNormal('v0', mu=mu_v0, sigma=sigma_v0,
                                       lower=1.5,
                                       shape=self.data['n_subjects'])
                g = pm.Gamma('g', alpha=4, beta=1.5 / mu_g,
                            shape=self.data['n_subjects'])

                stop_mu = pm.TruncatedNormal('stop_mu', mu=mu_stop_mu, sigma=sigma_stop_mu,
                                            lower=0.05, upper=0.5,
                                            shape=self.data['n_subjects'])
                stop_sigma = pm.HalfNormal('stop_sigma', sigma=mu_stop_sigma + sigma_stop_sigma,
                                          shape=self.data['n_subjects'])
                stop_tau = pm.HalfNormal('stop_tau', sigma=mu_stop_tau + sigma_stop_tau,
                                        shape=self.data['n_subjects'])

                pgf_probit = pm.Normal('pgf_probit', mu=mu_pgf_probit, sigma=sigma_pgf_probit,
                                      shape=self.data['n_subjects'])
                ptf_probit = pm.Normal('ptf_probit', mu=mu_ptf_probit, sigma=sigma_ptf_probit,
                                      shape=self.data['n_subjects'])

            else:
                # Individual-level model (no hierarchy)
                B = pm.Lognormal('B', mu=-0.2, sigma=0.3)
                t0 = pm.Beta('t0', alpha=3, beta=17)
                # CRITICAL FIX: Add lower bounds to prevent negative drift rates
                v_plus = pm.TruncatedNormal('v_plus', mu=3, sigma=0.5, lower=1.0)
                v_minus = pm.TruncatedNormal('v_minus', mu=0.8, sigma=0.3, lower=0.3)
                v0 = pm.TruncatedNormal('v0', mu=2.2, sigma=0.4, lower=1.5)
                g = pm.Gamma('g', alpha=4, beta=1.5)

                stop_mu = pm.TruncatedNormal('stop_mu', mu=0.25, sigma=0.1, lower=0.05, upper=0.5)
                stop_sigma = pm.HalfNormal('stop_sigma', sigma=0.1)
                stop_tau = pm.HalfNormal('stop_tau', sigma=0.1)

                pgf_probit = pm.Normal('pgf_probit', mu=-2, sigma=1)
                ptf_probit = pm.Normal('ptf_probit', mu=-2, sigma=1)

            # Convert probit to probabilities (ensures 0-1 range)
            pgf = pm.Deterministic('pgf', pm.math.invprobit(pgf_probit))
            ptf = pm.Deterministic('ptf', pm.math.invprobit(ptf_probit))

            # Compute SSRT from ex-Gaussian parameters (for monitoring)
            # SSRT = mu + tau (mean of ex-Gaussian, clamped at 0.05 lower bound)
            if self.use_hierarchical:
                ssrt = pm.Deterministic('ssrt',
                    pm.math.maximum(0.05, stop_mu + stop_tau))
            else:
                ssrt = pm.Deterministic('ssrt',
                    pm.math.maximum(0.05, stop_mu + stop_tau))

            # =================================================================
            # LIKELIHOOD FOR GO TRIALS
            # =================================================================
            if self.data['n_go'] > 0:
                # Get parameters for each go trial
                if self.use_hierarchical:
                    B_go = B[self.data['go_subject_idx']]
                    t0_go = t0[self.data['go_subject_idx']]
                    v_plus_go = v_plus[self.data['go_subject_idx']]
                    v_minus_go = v_minus[self.data['go_subject_idx']]
                    pgf_go = pgf[self.data['go_subject_idx']]
                else:
                    B_go = B
                    t0_go = t0
                    v_plus_go = v_plus
                    v_minus_go = v_minus
                    pgf_go = pgf

                # Drift rate depends on whether response matched stimulus
                v_go = pm.math.switch(
                    self.data['go_match'],
                    v_plus_go,  # Matching accumulator won
                    v_minus_go  # Mismatching accumulator won
                )

                # Decision time - ensure positive with max(0.01, ...)
                dt_go = pm.math.maximum(0.01, self.data['go_rt'] - t0_go)

                # Wald likelihood for go process
                # Ensure drift rate is positive
                v_go_safe = pm.math.maximum(0.1, v_go)
                mu_wald_go = B_go / v_go_safe
                lambda_wald_go = B_go ** 2

                logp_go = self._wald_logp(dt_go, mu_wald_go, lambda_wald_go)

                # Mixture with go failures (uniform contaminant process)
                # P(RT | no go failure) * (1 - pgf) + P(RT | go failure) * pgf
                # Using log-sum-exp for numerical stability
                logp_no_gf = logp_go + pm.math.log(1 - pgf_go)
                logp_gf = pm.math.log(pgf_go) + pm.math.log(1 / 2.0)  # Uniform over reasonable range

                logp_go_total = pm.math.logaddexp(logp_no_gf, logp_gf)

                pm.Potential('go_trials', pm.math.sum(logp_go_total))

            # =================================================================
            # LIKELIHOOD FOR SIGNAL-RESPOND TRIALS
            # =================================================================
            if self.data['n_signal_respond'] > 0:
                # Get parameters for each signal-respond trial
                if self.use_hierarchical:
                    B_sr = B[self.data['sr_subject_idx']]
                    t0_sr = t0[self.data['sr_subject_idx']]
                    v_plus_sr = v_plus[self.data['sr_subject_idx']]
                    v_minus_sr = v_minus[self.data['sr_subject_idx']]
                    v0_sr = v0[self.data['sr_subject_idx']]
                    g_sr = g[self.data['sr_subject_idx']]
                    stop_mu_sr = stop_mu[self.data['sr_subject_idx']]
                    stop_sigma_sr = stop_sigma[self.data['sr_subject_idx']]
                    stop_tau_sr = stop_tau[self.data['sr_subject_idx']]
                    ptf_sr = ptf[self.data['sr_subject_idx']]
                    pgf_sr = pgf[self.data['sr_subject_idx']]
                else:
                    B_sr = B
                    t0_sr = t0
                    v_plus_sr = v_plus
                    v_minus_sr = v_minus
                    v0_sr = v0
                    g_sr = g
                    stop_mu_sr = stop_mu
                    stop_sigma_sr = stop_sigma
                    stop_tau_sr = stop_tau
                    ptf_sr = ptf
                    pgf_sr = pgf

                # Compute SSD-dependent drift rates
                v_plus_at_ssd, v_minus_at_ssd = self._compute_drift_rates_for_ssd(
                    v_plus_sr, v_minus_sr, v0_sr, g_sr, self.data['sr_ssd']
                )

                # Drift rate depends on match
                v_sr = pm.math.switch(
                    self.data['sr_match'],
                    v_plus_at_ssd,
                    v_minus_at_ssd
                )

                # Decision time - ensure positive
                dt_sr = pm.math.maximum(0.01, self.data['sr_rt'] - t0_sr)

                # Go process likelihood
                # Ensure drift rate is positive
                v_sr_safe = pm.math.maximum(0.1, v_sr)
                mu_wald_sr = B_sr / v_sr_safe
                lambda_wald_sr = B_sr ** 2
                logp_go_sr = self._wald_logp(dt_sr, mu_wald_sr, lambda_wald_sr)

                # Stop process: signal-respond means stop process was slower
                # P(stop_time > dt_sr) = 1 - CDF(dt_sr - ssd)
                stop_time_threshold = dt_sr - self.data['sr_ssd']

                # Approximate survival function of ex-Gaussian
                # Using complement of CDF
                stop_cdf = self._exgaussian_cdf(
                    stop_time_threshold, stop_mu_sr, stop_sigma_sr, stop_tau_sr
                )
                # Clip CDF to valid range to prevent numerical issues
                stop_cdf = pm.math.clip(stop_cdf, 1e-10, 1 - 1e-10)
                logp_stop_slower = pm.math.log(1 - stop_cdf)

                # Combine: go process completed AND stop process was slower
                # With trigger failure: (1-ptf) * P(go wins) + ptf * P(go completes)
                logp_no_trigger_fail = pm.math.log(1 - ptf_sr) + logp_go_sr + logp_stop_slower
                logp_trigger_fail = pm.math.log(ptf_sr) + logp_go_sr

                logp_sr_total = pm.math.logaddexp(logp_no_trigger_fail, logp_trigger_fail)

                # Mixture with go failures
                logp_sr_no_gf = logp_sr_total + pm.math.log(1 - pgf_sr)
                logp_sr_gf = pm.math.log(pgf_sr) + pm.math.log(1 / 2.0)

                logp_sr_final = pm.math.logaddexp(logp_sr_no_gf, logp_sr_gf)

                pm.Potential('signal_respond_trials', pm.math.sum(logp_sr_final))

            # =================================================================
            # LIKELIHOOD FOR SUCCESSFUL STOP TRIALS
            # =================================================================
            if self.data['n_successful_stop'] > 0:
                # Get parameters
                if self.use_hierarchical:
                    B_ss = B[self.data['ss_subject_idx']]
                    t0_ss = t0[self.data['ss_subject_idx']]
                    v_plus_ss = v_plus[self.data['ss_subject_idx']]
                    v_minus_ss = v_minus[self.data['ss_subject_idx']]
                    v0_ss = v0[self.data['ss_subject_idx']]
                    g_ss = g[self.data['ss_subject_idx']]
                    stop_mu_ss = stop_mu[self.data['ss_subject_idx']]
                    stop_sigma_ss = stop_sigma[self.data['ss_subject_idx']]
                    stop_tau_ss = stop_tau[self.data['ss_subject_idx']]
                    ptf_ss = ptf[self.data['ss_subject_idx']]
                    pgf_ss = pgf[self.data['ss_subject_idx']]
                else:
                    B_ss = B
                    t0_ss = t0
                    v_plus_ss = v_plus
                    v_minus_ss = v_minus
                    v0_ss = v0
                    g_ss = g
                    stop_mu_ss = stop_mu
                    stop_sigma_ss = stop_sigma
                    stop_tau_ss = stop_tau
                    ptf_ss = ptf
                    pgf_ss = pgf

                # Compute SSD-dependent drift rates
                v_plus_at_ssd_ss, v_minus_at_ssd_ss = self._compute_drift_rates_for_ssd(
                    v_plus_ss, v_minus_ss, v0_ss, g_ss, self.data['ss_ssd']
                )

                # Average drift rate (we don't know which would have won)
                v_avg_ss = (v_plus_at_ssd_ss + v_minus_at_ssd_ss) / 2.0

                # Ensure positive drift rate
                v_avg_ss_safe = pm.math.maximum(0.1, v_avg_ss)

                # Expected go time distribution parameters
                mu_wald_ss = B_ss / v_avg_ss_safe
                lambda_wald_ss = B_ss ** 2

                # For successful stop, we need to integrate:
                # P(no response) = integral over all possible go times of:
                #   P(go_time = t) * P(stop_time < t - ssd)
                # This is computationally expensive, so we use approximation:
                # P(stop wins) ≈ P(stop_time < E[go_time] - ssd)

                expected_go_time = t0_ss + mu_wald_ss
                stop_time_to_beat = expected_go_time - self.data['ss_ssd']

                # CDF of stop time at this threshold
                stop_cdf_ss = self._exgaussian_cdf(
                    stop_time_to_beat, stop_mu_ss, stop_sigma_ss, stop_tau_ss
                )

                # Clip CDF to valid range
                stop_cdf_ss = pm.math.clip(stop_cdf_ss, 1e-10, 1 - 1e-10)

                # Probability stop wins (no trigger failure)
                logp_stop_wins = pm.math.log(stop_cdf_ss)

                # With trigger failure, can't stop
                logp_no_tf = pm.math.log(1 - ptf_ss) + logp_stop_wins

                # Successful stop is very unlikely if trigger failure
                # (only if go failure also occurs)
                logp_tf_and_gf = pm.math.log(ptf_ss) + pm.math.log(pgf_ss)

                logp_ss_total = pm.math.logaddexp(logp_no_tf, logp_tf_and_gf)

                pm.Potential('successful_stop_trials', pm.math.sum(logp_ss_total))

        self.model = model
        return model

    def fit(self, data_df: pd.DataFrame, draws: int = 1000, tune: int = 1000,
            chains: int = 4, cores: Optional[int] = None,
            **kwargs) -> az.InferenceData:
        """
        Fit the RDEX-ABCD model using PyMC's NUTS sampler.

        Parameters
        ----------
        data_df : pd.DataFrame
            Data with required columns
        draws : int, default=1000
            Number of posterior samples to draw
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
        print("Building RDEX-ABCD model...")
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
        """Print and return model summary."""
        if self.trace is None:
            print("Model has not been fitted yet.")
            return None

        summary_df = az.summary(self.trace)
        print(summary_df)
        return summary_df

    def plot_traces(self, var_names: Optional[list] = None, **kwargs):
        """Plot MCMC traces."""
        if self.trace is None:
            print("Model has not been fitted yet.")
            return

        if var_names is None:
            if self.use_hierarchical:
                var_names = ['mu_B', 'mu_t0', 'mu_v_plus', 'mu_v_minus',
                           'mu_v0', 'mu_g', 'mu_stop_mu', 'mu_pgf_probit', 'mu_ptf_probit']
            else:
                var_names = ['B', 't0', 'v_plus', 'v_minus', 'v0', 'g',
                           'stop_mu', 'ssrt', 'pgf', 'ptf']

        az.plot_trace(self.trace, var_names=var_names, **kwargs)
        plt.tight_layout()
        plt.show()

    def plot_posterior(self, var_names: Optional[list] = None, **kwargs):
        """Plot posterior distributions."""
        if self.trace is None:
            print("Model has not been fitted yet.")
            return

        az.plot_posterior(self.trace, var_names=var_names, **kwargs)
        plt.tight_layout()
        plt.show()

    def save_results(self, filepath: str):
        """Save results to NetCDF format."""
        if self.trace is None:
            raise ValueError("Model has not been fitted yet.")

        self.trace.to_netcdf(filepath)
        print(f"Results saved to: {filepath}")

    def load_results(self, filepath: str) -> az.InferenceData:
        """Load results from NetCDF format."""
        self.trace = az.from_netcdf(filepath)
        print(f"Results loaded from: {filepath}")
        return self.trace
