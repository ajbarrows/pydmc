"""
Example usage of the RDEX-ABCD model for ABCD stop-signal data.

This script demonstrates how to:
1. Load ABCD stop-signal data
2. Prepare it for the RDEX-ABCD model
3. Fit the model
4. Examine results
5. Extract key parameters (SSRT, trigger failure, processing speed, etc.)
"""

import pandas as pd
import numpy as np
from pydmc import RDEXABCDModel

# ==============================================================================
# 1. SIMULATE EXAMPLE DATA (replace with your actual ABCD data)
# ==============================================================================

def simulate_abcd_data(n_subjects=5, n_trials_per_subject=360):
    """
    Simulate ABCD-like stop-signal data for demonstration.

    In reality, you would load your actual ABCD data here.
    """
    np.random.seed(42)

    data = []
    for subj in range(n_subjects):
        # Individual differences in parameters
        v_plus = np.random.normal(3.3, 0.5)
        v_minus = np.random.normal(0.4, 0.3)
        v0 = np.random.normal(2.7, 0.4)
        g = np.random.gamma(3, 0.8)
        B = np.random.lognormal(0, 0.3) + 1.2
        t0 = np.random.beta(3, 17) + 0.1
        ssrt = np.random.normal(0.27, 0.03)

        n_go = int(n_trials_per_subject * 5/6)  # 83% go trials
        n_stop = n_trials_per_subject - n_go

        # Go trials
        for _ in range(n_go):
            stimulus = np.random.randint(0, 2)
            # Simulate choice and RT from racing diffusion
            if np.random.rand() < v_plus / (v_plus + v_minus):
                response = stimulus + 1  # Correct response
                rt = t0 + np.random.wald(B / v_plus, B**2)
            else:
                response = 2 - stimulus  # Error response
                rt = t0 + np.random.wald(B / v_minus, B**2)

            data.append({
                'subject': f'sub-{subj:03d}',
                'stimulus': stimulus,
                'response': response,
                'rt': min(rt, 1.5),  # Cap at 1.5s
                'ssd': np.nan
            })

        # Stop trials
        for _ in range(n_stop):
            stimulus = np.random.randint(0, 2)
            ssd = np.random.uniform(0.05, 0.45)

            # Compute SSD-dependent rates
            discrimination = min(g * ssd, v_plus - v0)
            v_plus_ssd = v0 + discrimination
            v_minus_ssd = v0 + max(discrimination - (v_plus - v_minus), 0)

            # Simulate go process
            if np.random.rand() < v_plus_ssd / (v_plus_ssd + v_minus_ssd):
                response_if_go = stimulus + 1
                rt_go = t0 + np.random.wald(B / v_plus_ssd, B**2)
            else:
                response_if_go = 2 - stimulus
                rt_go = t0 + np.random.wald(B / v_minus_ssd, B**2)

            # Simulate stop process
            rt_stop = ssd + np.random.normal(ssrt, 0.05)

            # Race: stop wins if rt_stop < rt_go
            if rt_stop < rt_go:
                response = 0  # Successful stop
                rt = np.nan
            else:
                response = response_if_go
                rt = rt_go

            data.append({
                'subject': f'sub-{subj:03d}',
                'stimulus': stimulus,
                'response': response,
                'rt': rt if not np.isnan(rt) else np.nan,
                'ssd': ssd
            })

    return pd.DataFrame(data)


# ==============================================================================
# 2. LOAD DATA
# ==============================================================================

print("=" * 70)
print("RDEX-ABCD Model Example")
print("=" * 70)
print()

# For this example, we simulate data
# In practice, replace this with:
# data = pd.read_csv('your_abcd_data.csv')
print("Loading data...")
data = simulate_abcd_data(n_subjects=5, n_trials_per_subject=360)

print(f"Loaded {len(data)} trials from {data['subject'].nunique()} subjects")
print(f"  - {data['ssd'].isna().sum()} go trials")
print(f"  - {(~data['ssd'].isna()).sum()} stop trials")
print()

# Show example data
print("Example data:")
print(data.head(10))
print()


# ==============================================================================
# 3. FIT THE RDEX-ABCD MODEL
# ==============================================================================

print("=" * 70)
print("Fitting RDEX-ABCD Model")
print("=" * 70)
print()

# Create model
print("Creating RDEX-ABCD model...")
model = RDEXABCDModel(use_hierarchical=True)
print("✓ Model created")
print()

# Fit model
# For a real analysis, use draws=1000, tune=1000 or more
# Here we use fewer for speed
print("Fitting model (this may take several minutes)...")
print("Note: For real analyses, use draws=1000+ and tune=1000+")
print()

trace = model.fit(
    data,
    draws=200,  # Increase for real analysis
    tune=200,   # Increase for real analysis
    chains=2,   # Increase to 4 for real analysis
    target_accept=0.9  # Higher acceptance rate for stability
)

print()
print("✓ Model fitting complete!")
print()


# ==============================================================================
# 4. EXAMINE RESULTS
# ==============================================================================

print("=" * 70)
print("Model Summary")
print("=" * 70)
print()

# Get summary statistics
summary = model.summary()
print()

# Plot traces (optional - comment out if running non-interactively)
# print("Plotting traces...")
# model.plot_traces()
# model.plot_posterior()


# ==============================================================================
# 5. EXTRACT KEY PARAMETERS
# ==============================================================================

print("=" * 70)
print("Key Parameters")
print("=" * 70)
print()

# Extract posterior means for key parameters
posterior = trace.posterior

print("GROUP-LEVEL PARAMETERS:")
print("-" * 70)

if 'mu_B' in posterior:
    print(f"Evidence threshold (mu_B):        {posterior['mu_B'].mean():.3f}")
    print(f"Non-decision time (mu_t0):        {posterior['mu_t0'].mean():.3f} s")
    print(f"Matching rate (mu_v_plus):        {posterior['mu_v_plus'].mean():.3f}")
    print(f"Mismatching rate (mu_v_minus):    {posterior['mu_v_minus'].mean():.3f}")
    print()
    print("CONTEXT INDEPENDENCE VIOLATION PARAMETERS:")
    print(f"Processing speed (mu_v0):         {posterior['mu_v0'].mean():.3f}")
    print(f"Perceptual growth rate (mu_g):    {posterior['mu_g'].mean():.3f}")
    print()
    print("STOP PROCESS PARAMETERS:")
    print(f"Stop ex-Gaussian μ (mu_stop_mu):  {posterior['mu_stop_mu'].mean():.3f} s")
    print(f"Mean SSRT (derived):               {(posterior['mu_stop_mu'] + posterior.get('mu_stop_tau', 0)).mean():.3f} s")
    print()
    print("FAILURE PROBABILITIES:")
    # Convert from probit scale
    import scipy.stats as stats
    mu_pgf_prob = stats.norm.cdf(posterior['mu_pgf_probit'].mean())
    mu_ptf_prob = stats.norm.cdf(posterior['mu_ptf_probit'].mean())
    print(f"Go failure probability (pgf):      {mu_pgf_prob:.3f}")
    print(f"Trigger failure probability (ptf): {mu_ptf_prob:.3f}")
else:
    print(f"Evidence threshold (B):            {posterior['B'].mean():.3f}")
    print(f"Non-decision time (t0):            {posterior['t0'].mean():.3f} s")
    print(f"Matching rate (v_plus):            {posterior['v_plus'].mean():.3f}")
    print(f"Mismatching rate (v_minus):        {posterior['v_minus'].mean():.3f}")
    print(f"Processing speed (v0):             {posterior['v0'].mean():.3f}")
    print(f"Perceptual growth rate (g):        {posterior['g'].mean():.3f}")
    print(f"SSRT:                              {posterior['ssrt'].mean():.3f} s")
    print(f"Go failure probability (pgf):      {posterior['pgf'].mean():.3f}")
    print(f"Trigger failure probability (ptf): {posterior['ptf'].mean():.3f}")

print()


# ==============================================================================
# 6. INTERPRETATION GUIDE
# ==============================================================================

print("=" * 70)
print("Interpretation Guide")
print("=" * 70)
print()

print("KEY INSIGHTS:")
print("-" * 70)
print()
print("1. SSRT (Stop-Signal Reaction Time):")
print("   - The main measure of inhibitory ability")
print("   - Lower values = better inhibition")
print("   - Typical values: 0.2 - 0.3 seconds in children")
print()
print("2. Trigger Failure (ptf):")
print("   - Proportion of trials where stop signal wasn't detected")
print("   - High values suggest attentional issues, not inhibitory deficits")
print("   - Critical to separate from true inhibition problems!")
print()
print("3. Processing Speed (v0):")
print("   - How fast evidence accumulates without discrimination")
print("   - High values = fast guessing, impulsive responding")
print("   - Affects accuracy on short-SSD trials")
print()
print("4. Perceptual Growth Rate (g):")
print("   - How quickly discriminative information accumulates with SSD")
print("   - High values = rapid perceptual processing")
print("   - Determines how quickly accuracy improves with presentation time")
print()
print("ADVANTAGES OF RDEX-ABCD:")
print("-" * 70)
print("✓ Accounts for ABCD task's context independence violation")
print("✓ Separates processing speed from inhibitory ability")
print("✓ Identifies trigger failures (attention lapses)")
print("✓ Provides unbiased SSRT estimates")
print("✓ Enables valid individual differences and group comparisons")
print()


# ==============================================================================
# 7. SAVE RESULTS
# ==============================================================================

print("=" * 70)
print("Saving Results")
print("=" * 70)
print()

# Save posterior samples
output_path = "rdex_abcd_results.nc"
model.save_results(output_path)
print(f"✓ Results saved to: {output_path}")
print()

# Save summary to CSV
summary_path = "rdex_abcd_summary.csv"
summary.to_csv(summary_path)
print(f"✓ Summary saved to: {summary_path}")
print()

print("=" * 70)
print("Analysis Complete!")
print("=" * 70)
print()
print("Next steps:")
print("1. Examine trace plots for convergence (Rhat < 1.01)")
print("2. Check posterior predictive plots")
print("3. Compare parameters across groups or conditions")
print("4. Link parameters to behavioral/neural outcomes")
print()
