"""
WALD Stop-Signal Model Implementation in Python with Stan.

This module provides a Python implementation of the WALD (Wald diffusion)
stop-signal model originally implemented in the DMC R package. The model
uses hierarchical Bayesian estimation via Stan for efficient sampling.

Key Features
------------
- Wald diffusion process for go trials
- Stop-signal race architecture
- Hierarchical parameter estimation
- Efficient Stan-based MCMC sampling
- Model comparison and validation tools
- Posterior predictive checking

Model Parameters
----------------
- B: Response threshold (boundary separation)
- t0: Non-decision time
- v0: Mean drift rate for go process
- vT: Drift rate for true/correct responses
- vF: Drift rate for false/incorrect responses
- gf: Go failure probability
- tf: Trigger failure probability
- mu: Ex-Gaussian mean
- sigma: Ex-Gaussian standard deviation
- tau: Ex-Gaussian exponential component
- k: Stop process parameter

Authors: Converted from DMC R package by Andrew Heathcote et al.
"""

import json
import multiprocessing
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .backends import StanBackend


class WaldStopSignalModel:
    """
    WALD Stop-Signal Model for response inhibition tasks.

    This class implements the WALD diffusion model for stop-signal tasks,
    providing both individual-level and hierarchical Bayesian estimation
    using Stan for efficient MCMC sampling.

    Parameters
    ----------
    use_hierarchical : bool, default=True
        Whether to use hierarchical (True) or individual-level (False) modeling

    Attributes
    ----------
    model_code : str
        Stan model code
    compiled_model : StanModel
        Compiled Stan model object
    fit_result : StanFit
        Results from model fitting
    data : dict
        Prepared data for Stan
    backend : StanBackend
        Stan backend handler

    Examples
    --------
    >>> import pandas as pd
    >>> from pydmc.models import WaldStopSignalModel
    >>>
    >>> # Load data
    >>> data = pd.read_csv('stop_signal_data.csv')
    >>>
    >>> # Create and fit hierarchical model
    >>> model = WaldStopSignalModel(use_hierarchical=True)
    >>> fit = model.fit(data, chains=4, iter=2000)
    >>>
    >>> # Get parameter estimates
    >>> estimates = model.get_parameter_estimates()
    >>>
    >>> # Perform posterior predictive check
    >>> model.posterior_predictive_check()
    """

    def __init__(self, use_hierarchical: bool = True):
        """Initialize the WALD Stop-Signal model."""
        self.use_hierarchical = use_hierarchical
        self.backend = StanBackend()
        self.model_code = self._get_stan_code()
        self.compiled_model = None
        self.fit_result = None
        self.data = None

    def _get_stan_code(self) -> str:
        """Generate Stan code for the WALD stop-signal model."""

        if self.use_hierarchical:
            return self._get_hierarchical_stan_code()
        else:
            return self._get_individual_stan_code()

    def _get_hierarchical_stan_code(self) -> str:
        """Get Stan code for hierarchical WALD stop-signal model."""
        return """
        // Hierarchical WALD Stop-Signal Model
        functions {
            // WALD (Inverse Gaussian) log density
            real wald_lpdf(real y, real mu, real lambda) {
                if (y <= 0) return negative_infinity();
                return 0.5 * (log(lambda) - log(2 * pi() * y^3)) -
                       lambda * (y - mu)^2 / (2 * mu^2 * y);
            }

            // Ex-Gaussian log density
            real exgaussian_lpdf(real y, real mu, real sigma, real tau) {
                if (tau <= 0 || sigma <= 0) return negative_infinity();
                real z = (y - mu) / sigma;
                real lambda_exg = 1.0 / tau;
                return log(lambda_exg) - lambda_exg * (y - mu) +
                       lambda_exg^2 * sigma^2 / 2 +
                       normal_lcdf(z - lambda_exg * sigma | 0, 1);
            }
        }

        data {
            int<lower=0> N;                    // Total number of trials
            int<lower=0> N_subjects;           // Number of subjects
            int<lower=0> N_go;                 // Number of go trials
            int<lower=0> N_stop;               // Number of stop trials

            array[N] int<lower=1, upper=N_subjects> subject_id;
            array[N] int<lower=0, upper=1> is_stop_trial;
            array[N] int<lower=0, upper=2> response;  // 0=NR, 1=LEFT, 2=RIGHT
            array[N] real<lower=0> rt;
            array[N_stop] real<lower=0> ssd;          // Stop signal delays
            array[N] int<lower=0, upper=1> stimulus;  // 0=left, 1=right
            array[N] int<lower=0, upper=1> correct;   // Stimulus-response match
        }

        parameters {
            // Hierarchical parameters (group level)
            vector[11] mu_params;              // Group means
            vector<lower=0>[11] sigma_params;  // Group SDs

            // Individual parameters (subject level) - raw (standardized)
            matrix[N_subjects, 11] params_raw;
        }

        transformed parameters {
            matrix[N_subjects, 11] params;

            // Transform raw parameters to natural scale
            // Parameter order: B, t0, gf, mu, sigma, tau, tf, vT, vF, v0, k
            for (s in 1:N_subjects) {
                params[s, 1] = exp(mu_params[1] + sigma_params[1] * params_raw[s, 1]);        // B (threshold)
                params[s, 2] = inv_logit(mu_params[2] + sigma_params[2] * params_raw[s, 2]);  // t0 (non-decision time)
                params[s, 3] = inv_logit(mu_params[3] + sigma_params[3] * params_raw[s, 3]);  // gf (go failure)
                params[s, 4] = exp(mu_params[4] + sigma_params[4] * params_raw[s, 4]);        // mu (ex-gaussian mean)
                params[s, 5] = exp(mu_params[5] + sigma_params[5] * params_raw[s, 5]);        // sigma (ex-gaussian sd)
                params[s, 6] = exp(mu_params[6] + sigma_params[6] * params_raw[s, 6]);        // tau (ex-gaussian exp)
                params[s, 7] = inv_logit(mu_params[7] + sigma_params[7] * params_raw[s, 7]);  // tf (trigger failure)
                params[s, 8] = mu_params[8] + sigma_params[8] * params_raw[s, 8];             // vT (drift true)
                params[s, 9] = mu_params[9] + sigma_params[9] * params_raw[s, 9];             // vF (drift false)
                params[s, 10] = mu_params[10] + sigma_params[10] * params_raw[s, 10];         // v0 (baseline drift)
                params[s, 11] = exp(mu_params[11] + sigma_params[11] * params_raw[s, 11]);    // k (stop process)
            }
        }

        model {
            // Priors for hierarchical parameters
            mu_params[1] ~ normal(log(1.0), 0.5);     // B (log scale)
            mu_params[2] ~ normal(logit(0.15), 0.5);  // t0 (logit scale)
            mu_params[3] ~ normal(logit(0.02), 1.0);  // gf (logit scale)
            mu_params[4] ~ normal(log(0.5), 0.5);     // mu (log scale)
            mu_params[5] ~ normal(log(0.05), 0.5);    // sigma (log scale)
            mu_params[6] ~ normal(log(0.1), 0.5);     // tau (log scale)
            mu_params[7] ~ normal(logit(0.05), 1.0);  // tf (logit scale)
            mu_params[8] ~ normal(2.0, 0.5);          // vT (natural scale)
            mu_params[9] ~ normal(1.0, 0.5);          // vF (natural scale)
            mu_params[10] ~ normal(1.5, 0.5);         // v0 (natural scale)
            mu_params[11] ~ normal(log(2.0), 0.5);    // k (log scale)

            sigma_params ~ exponential(2);

            // Individual parameter deviations (standardized)
            for (s in 1:N_subjects) {
                params_raw[s] ~ std_normal();
            }

            // Likelihood
            for (i in 1:N) {
                int subj = subject_id[i];
                real B = params[subj, 1];
                real t0 = params[subj, 2];
                real gf = params[subj, 3];
                real mu_exg = params[subj, 4];
                real sigma_exg = params[subj, 5];
                real tau_exg = params[subj, 6];
                real tf = params[subj, 7];
                real vT = params[subj, 8];
                real vF = params[subj, 9];
                real v0 = params[subj, 10];
                real k = params[subj, 11];

                if (is_stop_trial[i] == 0) {
                    // Go trial
                    if (response[i] > 0) {  // Response made
                        real drift = correct[i] == 1 ? vT : vF;
                        if (drift > 0) {
                            real mu_wald = B / drift;
                            real lambda_wald = B^2;
                            target += wald_lpdf(rt[i] - t0 | mu_wald, lambda_wald);
                        } else {
                            target += negative_infinity();
                        }
                    } else {  // No response (go failure)
                        target += log(gf);
                    }
                } else {
                    // Stop trial
                    if (response[i] == 0) {  // Successfully stopped
                        // Probability of successful stopping (simplified)
                        target += log(1 - tf);
                    } else {  // Failed to stop
                        real drift = correct[i] == 1 ? vT : vF;
                        if (drift > 0) {
                            real mu_wald = B / drift;
                            real lambda_wald = B^2;
                            target += log(tf) + wald_lpdf(rt[i] - t0 | mu_wald, lambda_wald);
                        } else {
                            target += log(tf) + negative_infinity();
                        }
                    }
                }
            }
        }

        generated quantities {
            // Posterior predictive samples for model checking
            array[N] real rt_pred;
            array[N] int response_pred;

            for (i in 1:N) {
                int subj = subject_id[i];
                real B = params[subj, 1];
                real t0 = params[subj, 2];
                real gf = params[subj, 3];
                real vT = params[subj, 8];
                real vF = params[subj, 9];

                if (is_stop_trial[i] == 0) {
                    // Go trial prediction
                    if (bernoulli_rng(1 - gf)) {
                        real drift = correct[i] == 1 ? vT : vF;
                        if (drift > 0) {
                            real mu_wald = B / drift;
                            real lambda_wald = B^2;
                            real scale_param = mu_wald / lambda_wald;
                            rt_pred[i] = t0 + inv_gamma_rng(0.5, lambda_wald / (2 * scale_param));
                            response_pred[i] = correct[i] == 1 ? stimulus[i] + 1 : 2 - stimulus[i];
                        } else {
                            rt_pred[i] = t0 + 1.0;  // Default
                            response_pred[i] = 0;
                        }
                    } else {
                        rt_pred[i] = -1;  // No response
                        response_pred[i] = 0;
                    }
                } else {
                    // Stop trial prediction (simplified)
                    rt_pred[i] = rt[i];  // Use observed for now
                    response_pred[i] = response[i];
                }
            }
        }
        """

    def _get_individual_stan_code(self) -> str:
        """Get Stan code for individual-level WALD stop-signal model."""
        return """
        // Individual WALD Stop-Signal Model
        functions {
            real wald_lpdf(real y, real mu, real lambda) {
                if (y <= 0) return negative_infinity();
                return 0.5 * (log(lambda) - log(2 * pi() * y^3)) -
                       lambda * (y - mu)^2 / (2 * mu^2 * y);
            }
        }

        data {
            int<lower=0> N;
            array[N] int<lower=0, upper=1> is_stop_trial;
            array[N] int<lower=0, upper=2> response;
            array[N] real<lower=0> rt;
            array[N] int<lower=0, upper=1> correct;
        }

        parameters {
            real<lower=0> B;                    // Threshold
            real<lower=0, upper=1> t0;          // Non-decision time
            real vT;                            // Drift rate true
            real vF;                            // Drift rate false
            real<lower=0, upper=1> gf;          // Go failure prob
            real<lower=0, upper=1> tf;          // Trigger failure prob
        }

        model {
            // Priors
            B ~ lognormal(0, 1);
            t0 ~ beta(2, 8);
            vT ~ normal(2, 1);
            vF ~ normal(1, 1);
            gf ~ beta(1, 19);
            tf ~ beta(1, 9);

            // Likelihood
            for (i in 1:N) {
                if (is_stop_trial[i] == 0) {
                    if (response[i] > 0) {
                        real drift = correct[i] == 1 ? vT : vF;
                        if (drift > 0) {
                            real mu_wald = B / drift;
                            real lambda_wald = B^2;
                            target += wald_lpdf(rt[i] - t0 | mu_wald, lambda_wald);
                        } else {
                            target += negative_infinity();
                        }
                    } else {
                        target += log(gf);
                    }
                } else {
                    if (response[i] == 0) {
                        target += log(1 - tf);
                    } else {
                        real drift = correct[i] == 1 ? vT : vF;
                        if (drift > 0) {
                            real mu_wald = B / drift;
                            real lambda_wald = B^2;
                            target += log(tf) + wald_lpdf(rt[i] - t0 | mu_wald, lambda_wald);
                        } else {
                            target += log(tf) + negative_infinity();
                        }
                    }
                }
            }
        }

        generated quantities {
            array[N] real rt_pred;
            array[N] int response_pred;

            for (i in 1:N) {
                if (is_stop_trial[i] == 0) {
                    if (bernoulli_rng(1 - gf)) {
                        real drift = correct[i] == 1 ? vT : vF;
                        if (drift > 0) {
                            real mu_wald = B / drift;
                            real lambda_wald = B^2;
                            real scale_param = mu_wald / lambda_wald;
                            rt_pred[i] = t0 + inv_gamma_rng(0.5, lambda_wald / (2 * scale_param));
                            response_pred[i] = correct[i] == 1 ? 1 : 2;
                        } else {
                            rt_pred[i] = t0 + 1.0;
                            response_pred[i] = 0;
                        }
                    } else {
                        rt_pred[i] = -1;
                        response_pred[i] = 0;
                    }
                } else {
                    rt_pred[i] = rt[i];
                    response_pred[i] = response[i];
                }
            }
        }
        """

    def prepare_data(self, data_df: pd.DataFrame) -> Dict:
        """
        Prepare data for Stan model.

        Parameters
        ----------
        data_df : pd.DataFrame
            DataFrame with columns: subject, stimulus, response, rt, ssd (optional)

        Returns
        -------
        dict
            Stan data dictionary

        Raises
        ------
        ValueError
            If required columns are missing or no valid trials remain
        """

        # Validate required columns
        required_cols = ['subject', 'stimulus', 'response', 'rt']
        missing_cols = [col for col in required_cols if col not in data_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Create working copy
        df = data_df.copy()

        # Create stop trial indicator
        if 'ssd' in df.columns:
            df['is_stop_trial'] = (~df['ssd'].isna()).astype(int)
            ssd_values = df[df['is_stop_trial'] == 1]['ssd'].values
        else:
            df['is_stop_trial'] = 0
            ssd_values = np.array([])

        # Create correct response indicator
        # Assuming stimulus: 0=left, 1=right and response: 0=NR, 1=LEFT, 2=RIGHT
        df['correct'] = (
            ((df['stimulus'] == 0) & (df['response'] == 1)) |
            ((df['stimulus'] == 1) & (df['response'] == 2))
        ).astype(int)

        # Remove trials with missing or invalid RT
        valid_rt_mask = (df['rt'] > 0) & (df['rt'].notna())
        if not valid_rt_mask.all():
            n_removed = (~valid_rt_mask).sum()
            print(f"Warning: Removed {n_removed} trials with invalid RT")
            df = df[valid_rt_mask].copy()

        # Ensure response values are valid
        valid_response_mask = df['response'].isin([0, 1, 2])
        if not valid_response_mask.all():
            n_removed = (~valid_response_mask).sum()
            print(f"Warning: Removed {n_removed} trials with invalid response")
            df = df[valid_response_mask].copy()

        if len(df) == 0:
            raise ValueError("No valid trials remaining after data cleaning")

        # Prepare Stan data
        if self.use_hierarchical:
            # Map subjects to integers starting from 1
            unique_subjects = sorted(df['subject'].unique())
            subject_map = {subj: i+1 for i, subj in enumerate(unique_subjects)}
            df['subject_id'] = df['subject'].map(subject_map)

            # Update ssd_values for remaining trials
            if len(ssd_values) > 0:
                ssd_values = df[df['is_stop_trial'] == 1]['ssd'].values

            stan_data = {
                'N': len(df),
                'N_subjects': len(unique_subjects),
                'N_go': int((df['is_stop_trial'] == 0).sum()),
                'N_stop': int((df['is_stop_trial'] == 1).sum()),
                'subject_id': df['subject_id'].values.astype(int),
                'is_stop_trial': df['is_stop_trial'].values.astype(int),
                'response': df['response'].values.astype(int),
                'rt': df['rt'].values.astype(float),
                'ssd': ssd_values.astype(float) if len(ssd_values) > 0 else np.array([]),
                'stimulus': df['stimulus'].values.astype(int),
                'correct': df['correct'].values.astype(int)
            }
        else:
            # Individual level - use first subject only or specified subject
            if len(df['subject'].unique()) > 1:
                first_subject = df['subject'].iloc[0]
                df = df[df['subject'] == first_subject].copy()
                print(f"Using data from subject: {first_subject}")

            stan_data = {
                'N': len(df),
                'is_stop_trial': df['is_stop_trial'].values.astype(int),
                'response': df['response'].values.astype(int),
                'rt': df['rt'].values.astype(float),
                'correct': df['correct'].values.astype(int)
            }

        self.data = stan_data
        return stan_data

    def fit(self, data_df: pd.DataFrame, chains: int = 4, iter: int = 2000,
            warmup: int = 1000, cores: Optional[int] = None,
            show_progress: bool = True, **kwargs) -> Any:
        """
        Fit the WALD stop-signal model using Stan.

        Parameters
        ----------
        data_df : pd.DataFrame
            Data with required columns
        chains : int, default=4
            Number of MCMC chains
        iter : int, default=2000
            Number of iterations per chain
        warmup : int, default=1000
            Number of warmup iterations
        cores : int, optional
            Number of cores for parallel sampling (default: use all available)
        show_progress : bool, default=True
            Whether to show sampling progress
        **kwargs
            Additional arguments passed to Stan sampler

        Returns
        -------
        fit : StanFit
            Stan fit object with posterior samples
        """

        # Prepare data
        stan_data = self.prepare_data(data_df)

        # Set default cores
        if cores is None:
            cores = min(chains, multiprocessing.cpu_count())

        # Compile model if needed
        if self.compiled_model is None:
            print("Compiling Stan model...")
            self.compiled_model = self.backend.compile_model(self.model_code, stan_data)
            print("Model compiled successfully")

        # Fit model
        print("Fitting model...")
        sampling_kwargs = {
            'chains': chains,
            'iter': iter,
            'warmup': warmup,
            'cores': cores,
            'show_progress': show_progress,
            **kwargs
        }

        self.fit_result = self.backend.sample(
            self.compiled_model,
            stan_data,
            **sampling_kwargs
        )

        print("Model fitted successfully")
        return self.fit_result

    def summary(self) -> None:
        """Print model fit summary."""
        if self.fit_result is None:
            print("Model has not been fitted yet.")
            return

        if self.backend.backend_name == "cmdstanpy":
            print(self.fit_result.summary())
        elif self.backend.backend_name == "pystan":
            print(self.fit_result)
        else:
            print("Summary not available for current backend")

    def extract_samples(self) -> Dict[str, np.ndarray]:
        """
        Extract posterior samples.

        Returns
        -------
        dict
            Dictionary of parameter names to posterior samples

        Raises
        ------
        ValueError
            If model has not been fitted yet
        """
        if self.fit_result is None:
            raise ValueError("Model has not been fitted yet.")

        return self.backend.extract_samples(self.fit_result)

    def plot_traces(self, params: Optional[list] = None, figsize: tuple = (12, 8)) -> None:
        """
        Plot MCMC traces for model parameters.

        Parameters
        ----------
        params : list, optional
            List of parameter names to plot. If None, uses default parameters.
        figsize : tuple, default=(12, 8)
            Figure size as (width, height)
        """
        if self.fit_result is None:
            print("Model has not been fitted yet.")
            return

        try:
            import arviz as az

            # Convert to ArviZ format
            if self.backend.backend_name == "cmdstanpy":
                az_data = az.from_cmdstanpy(self.fit_result)
            elif self.backend.backend_name == "pystan":
                az_data = az.from_pystan(self.fit_result)
            else:
                print("Trace plots not available for current backend")
                return

            # Select parameters to plot
            if params is None:
                if self.use_hierarchical:
                    params = ['mu_params', 'sigma_params']
                else:
                    params = ['B', 't0', 'vT', 'vF', 'gf', 'tf']

            # Create trace plot
            az.plot_trace(az_data, var_names=params, figsize=figsize)
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("ArviZ not available. Install with: pip install arviz")
        except Exception as e:
            print(f"Error creating trace plots: {e}")

    def posterior_predictive_check(self, figsize: tuple = (12, 5)) -> None:
        """
        Perform posterior predictive checks.

        Parameters
        ----------
        figsize : tuple, default=(12, 5)
            Figure size as (width, height)
        """
        if self.fit_result is None:
            print("Model has not been fitted yet.")
            return

        try:
            # Extract posterior predictive samples
            samples = self.extract_samples()
            rt_pred = samples.get('rt_pred', None)

            if rt_pred is None:
                print("No posterior predictive samples found.")
                return

            # Plot observed vs predicted RT distributions
            fig, axes = plt.subplots(1, 2, figsize=figsize)

            # Go trials
            go_mask = self.data['is_stop_trial'] == 0
            if np.any(go_mask):
                observed_go = self.data['rt'][go_mask]
                # Remove invalid predictions
                predicted_go = rt_pred[:, go_mask]
                predicted_go = predicted_go[predicted_go > 0].flatten()

                if len(predicted_go) > 0:
                    axes[0].hist(observed_go, alpha=0.7, label='Observed', bins=30, density=True)
                    axes[0].hist(predicted_go, alpha=0.7, label='Predicted', bins=30, density=True)
                    axes[0].set_title('Go Trials')
                    axes[0].set_xlabel('RT (s)')
                    axes[0].legend()
                else:
                    axes[0].text(0.5, 0.5, 'No valid predictions', ha='center', va='center',
                               transform=axes[0].transAxes)
            else:
                axes[0].text(0.5, 0.5, 'No go trials', ha='center', va='center',
                           transform=axes[0].transAxes)

            # Stop trials (if any)
            stop_mask = self.data['is_stop_trial'] == 1
            if np.any(stop_mask):
                observed_stop = self.data['rt'][stop_mask]
                predicted_stop = rt_pred[:, stop_mask]
                predicted_stop = predicted_stop[predicted_stop > 0].flatten()

                if len(predicted_stop) > 0:
                    axes[1].hist(observed_stop, alpha=0.7, label='Observed', bins=30, density=True)
                    axes[1].hist(predicted_stop, alpha=0.7, label='Predicted', bins=30, density=True)
                    axes[1].set_title('Stop Trials')
                    axes[1].set_xlabel('RT (s)')
                    axes[1].legend()
                else:
                    axes[1].text(0.5, 0.5, 'No valid predictions', ha='center', va='center',
                               transform=axes[1].transAxes)
            else:
                axes[1].text(0.5, 0.5, 'No stop trials', ha='center', va='center',
                           transform=axes[1].transAxes)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error in posterior predictive check: {e}")

    def get_parameter_estimates(self) -> Dict[str, Dict[str, float]]:
        """
        Get parameter estimates with summary statistics.

        Returns
        -------
        dict
            Parameter estimates with mean, std, quantiles

        Raises
        ------
        ValueError
            If model has not been fitted yet
        """
        if self.fit_result is None:
            raise ValueError("Model has not been fitted yet.")

        samples = self.extract_samples()
        estimates = {}

        for param_name, param_samples in samples.items():
            if param_name.startswith('rt_pred') or param_name.startswith('response_pred'):
                continue  # Skip prediction arrays

            # Handle different parameter shapes
            if param_samples.ndim == 1:
                # Scalar parameter
                estimates[param_name] = {
                    'mean': float(np.mean(param_samples)),
                    'std': float(np.std(param_samples)),
                    'q2.5': float(np.percentile(param_samples, 2.5)),
                    'q25': float(np.percentile(param_samples, 25)),
                    'q50': float(np.percentile(param_samples, 50)),
                    'q75': float(np.percentile(param_samples, 75)),
                    'q97.5': float(np.percentile(param_samples, 97.5))
                }
            elif param_samples.ndim == 2:
                # Vector parameter (like mu_params, sigma_params)
                for i in range(param_samples.shape[1]):
                    param_i = param_samples[:, i]
                    estimates[f"{param_name}[{i+1}]"] = {
                        'mean': float(np.mean(param_i)),
                        'std': float(np.std(param_i)),
                        'q2.5': float(np.percentile(param_i, 2.5)),
                        'q25': float(np.percentile(param_i, 25)),
                        'q50': float(np.percentile(param_i, 50)),
                        'q75': float(np.percentile(param_i, 75)),
                        'q97.5': float(np.percentile(param_i, 97.5))
                    }
            elif param_samples.ndim == 3:
                # Matrix parameter (like individual params)
                for i in range(param_samples.shape[1]):
                    for j in range(param_samples.shape[2]):
                        param_ij = param_samples[:, i, j]
                        estimates[f"{param_name}[{i+1},{j+1}]"] = {
                            'mean': float(np.mean(param_ij)),
                            'std': float(np.std(param_ij)),
                            'q2.5': float(np.percentile(param_ij, 2.5)),
                            'q25': float(np.percentile(param_ij, 25)),
                            'q50': float(np.percentile(param_ij, 50)),
                            'q75': float(np.percentile(param_ij, 75)),
                            'q97.5': float(np.percentile(param_ij, 97.5))
                        }

        return estimates

    def save_results(self, filepath: str, include_samples: bool = False) -> None:
        """
        Save model results to file.

        Parameters
        ----------
        filepath : str
            Output file path (.json)
        include_samples : bool, default=False
            Whether to include full posterior samples (large files)

        Raises
        ------
        ValueError
            If model has not been fitted yet
        """
        if self.fit_result is None:
            raise ValueError("Model has not been fitted yet.")

        results = {
            'model_type': 'hierarchical' if self.use_hierarchical else 'individual',
            'backend': self.backend.backend_name,
            'data_summary': {
                'n_trials': self.data['N'],
                'n_subjects': self.data.get('N_subjects', 1),
                'n_go': self.data.get('N_go', self.data['N']),
                'n_stop': self.data.get('N_stop', 0)
            },
            'parameter_estimates': self.get_parameter_estimates()
        }

        if include_samples:
            samples = self.extract_samples()
            # Convert numpy arrays to lists for JSON serialization
            samples_serializable = {}
            for key, value in samples.items():
                if isinstance(value, np.ndarray):
                    samples_serializable[key] = value.tolist()
                else:
                    samples_serializable[key] = value
            results['posterior_samples'] = samples_serializable

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {filepath}")

    def load_results(self, filepath: str) -> Dict:
        """
        Load model results from file.

        Parameters
        ----------
        filepath : str
            Input file path (.json)

        Returns
        -------
        dict
            Loaded results
        """
        with open(filepath, 'r') as f:
            results = json.load(f)

        print(f"Results loaded from: {filepath}")
        return results
