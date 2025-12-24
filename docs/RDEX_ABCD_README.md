# RDEX-ABCD Model Implementation

This directory contains a PyMC implementation of the **RDEX-ABCD model** from:

> Weigard, A., Matzke, D., Tanis, C., & Heathcote, A. (2023). A cognitive process modeling framework for the ABCD study stop-signal task. *Developmental Cognitive Neuroscience, 59*, 101191.

## Overview

The RDEX-ABCD model is specifically designed for the ABCD Study's stop-signal task, which has a unique design feature: the visual stop signal **replaces** the go stimulus, violating the standard "context independence" assumption. This creates a critical measurement problem that the RDEX-ABCD model solves.

## The Problem: Context Independence Violation

In the ABCD task:
- **Go trials**: Choice stimulus presented for up to 1 second
- **Stop trials**: Stop signal *replaces* the choice stimulus at SSD
- **Result**: At short SSDs, participants have limited time to process the choice stimulus

This means:
- At SSD = 0: No choice information available → chance accuracy
- At SSD ~ 0.3-0.4s: Full choice information available → normal accuracy

**This violates context independence** because the go process is NOT the same on stop trials vs. go trials!

## The Solution: RDEX-ABCD Model

The RDEX-ABCD model accounts for this by:

### 1. **Racing Diffusion for Go Process**
Two evidence accumulators race to threshold:
- **Matching accumulator**: Favors the correct response (drift rate `v+`)
- **Mismatching accumulator**: Favors the incorrect response (drift rate `v-`)

### 2. **Context Independence Violation Parameters**

**Processing Speed (`v0`)**:
- Base evidence accumulation rate when no discriminative information available
- Active at SSD = 0 (no stimulus)
- Drives fast guessing responses

**Perceptual Growth Rate (`g`)**:
- Rate at which discriminative information grows with SSD
- At SSD = 0: both accumulators have rate `v0`
- As SSD increases: rates grow linearly until reaching `v+` and `v-`

**Mathematical Model**:
```
At SSD = 0:
  matching_rate = v0
  mismatching_rate = v0

As SSD increases:
  discrimination_growth = min(g * SSD, asymptotic_value)
  matching_rate = v0 + discrimination_growth_to_v_plus
  mismatching_rate = v0 + discrimination_growth_to_v_minus

At long SSD (asymptote):
  matching_rate = v+
  mismatching_rate = v-
```

### 3. **Ex-Gaussian Stop Process**
Stop process finishing times follow an ex-Gaussian distribution:
- `μ` (mu): Normal component mean
- `σ` (sigma): Normal component SD
- `τ` (tau): Exponential component mean
- **SSRT = μ + τ** (mean stop-signal reaction time)

### 4. **Failure Processes**

**Trigger Failure (`ptf`)**:
- Probability of failing to detect the stop signal
- Leads to signal-respond trials even at easy SSDs

**Go Failure (`pgf`)**:
- Probability of failing to respond on go trials
- Accounts for omissions

## Complete Parameter List

| Parameter | Description | Type |
|-----------|-------------|------|
| `t0` | Non-decision time (encoding + motor) | Go process |
| `B` | Evidence threshold | Go process |
| `v+` | Matching drift rate (asymptotic) | Go process |
| `v-` | Mismatching drift rate (asymptotic) | Go process |
| `v0` | Processing speed (no discrimination) | **Context violation** |
| `g` | Perceptual growth rate | **Context violation** |
| `pgf` | Probability of go failure | Go process |
| `μ` | Stop ex-Gaussian normal mean | Stop process |
| `σ` | Stop ex-Gaussian normal SD | Stop process |
| `τ` | Stop ex-Gaussian exponential mean | Stop process |
| `ptf` | Probability of trigger failure | Stop process |
| **SSRT** | μ + τ (derived) | **Main measure** |

## Usage

### Basic Usage

```python
import pandas as pd
from pydmc import RDEXABCDModel

# Load ABCD stop-signal data
# Required columns: subject, stimulus, response, rt, ssd
data = pd.read_csv('abcd_data.csv')

# Create model
model = RDEXABCDModel(use_hierarchical=True)

# Fit model
trace = model.fit(
    data,
    draws=1000,
    tune=1000,
    chains=4
)

# View results
model.summary()
model.plot_traces()
model.plot_posterior()
```

### Understanding the Output

Key parameters to examine:

```python
# SSRT - the main measure of inhibitory ability
# Lower = better inhibition
ssrt = trace.posterior['ssrt']

# Trigger failure - attention to stop signal
# Higher = more attention lapses
ptf = trace.posterior['ptf']

# Processing speed - fast guessing
# Higher = faster processing even without discrimination
v0 = trace.posterior['v0']

# Perceptual growth rate - how fast discrimination improves with SSD
# Higher = faster accumulation of choice information
g = trace.posterior['g']
```

## Model Comparison: RDEX-ABCD vs. Simple Models

| Feature | Simple Wald Model | RDEX-ABCD Model |
|---------|-------------------|-----------------|
| Go process | Simple Wald | Racing diffusion |
| Stop process | Ex-Gaussian ✓ | Ex-Gaussian ✓ |
| Context independence | Assumed | **Explicitly modeled** |
| Trigger failure | ✗ | ✓ |
| Go failure | ✗ | ✓ |
| SSD-dependent rates | ✗ | **✓ (key innovation)** |
| Choice accuracy | Not modeled | **Fully modeled** |
| Suitable for ABCD | ⚠️ Biased | ✓ Unbiased |

## Why This Matters for ABCD

### Without RDEX-ABCD:
- **Biased SSRT estimates** due to context violations
- **Confounded effects**: Processing speed differences can be mistaken for inhibitory differences
- **Wrong conclusions**: Can reverse the order of who has better inhibition

### With RDEX-ABCD:
- ✓ Unbiased SSRT estimates
- ✓ Separate measures of processing speed vs. inhibition
- ✓ Accurate individual differences
- ✓ Valid group comparisons
- ✓ Additional mechanistic insights (trigger failure, processing speed, perceptual growth)

## Data Requirements

Your DataFrame must include:

```python
Required columns:
- 'subject': Subject identifier
- 'stimulus': Stimulus type (0 or 1, e.g., left=0, right=1)
- 'response': Response (0=no response, 1=left, 2=right, or similar)
- 'rt': Response time in seconds
- 'ssd': Stop-signal delay in seconds (NaN for go trials)

Example:
   subject  stimulus  response    rt   ssd
0        1         0         1  0.45   NaN   # Go trial, correct
1        1         1         2  0.52   NaN   # Go trial, correct
2        1         0         1  0.48  0.25   # Stop trial, failed to stop
3        1         1         0  NaN   0.30   # Stop trial, successful stop
```

## Interpretation Guide

### High `v0` (Processing Speed)
- Fast responses even without discriminative information
- May indicate impulsive responding or lower threshold for action

### High `g` (Perceptual Growth Rate)
- Rapidly accumulates discriminative information
- Efficient perceptual processing
- Choice accuracy improves quickly with presentation time

### High `ptf` (Trigger Failure)
- Frequent attention lapses
- May indicate ADHD, inattention, or task disengagement
- **Critical**: Can masquerade as poor inhibition in standard analyses!

### High SSRT (μ + τ)
- Slower inhibitory process
- True inhibitory deficit (after accounting for trigger failures)

## Technical Details

### Likelihood Components

The full model likelihood includes three trial types:

**1. Go Trials**:
```
P(RT, response) = (1 - pgf) * P(RT | racing diffusion) +
                  pgf * P(RT | uniform contaminant)
```

**2. Signal-Respond Trials**:
```
P(RT, response | SSD) =
  [(1 - ptf) * P(go wins race with stop | SSD-dependent rates) +
   ptf * P(go completes)] * (1 - pgf) +
  pgf * P(contaminant)
```

**3. Successful Stop Trials**:
```
P(no response | SSD) =
  (1 - ptf) * P(stop wins race | SSD-dependent rates) +
  ptf * pgf
```

### SSD-Dependent Drift Rates

On stop trials, drift rates are computed as:

```python
def compute_rates(v_plus, v_minus, v0, g, ssd):
    # Discrimination components
    match_disc = min(g * ssd, v_plus - v0)
    mismatch_disc = min(g * ssd, v_minus - v0)

    # Final rates
    match_rate = v0 + match_disc
    mismatch_rate = v0 + mismatch_disc

    return match_rate, mismatch_rate
```

## References

### Primary Reference
Weigard, A., Matzke, D., Tanis, C., & Heathcote, A. (2023). A cognitive process modeling framework for the ABCD study stop-signal task. *Developmental Cognitive Neuroscience, 59*, 101191. https://doi.org/10.1016/j.dcn.2022.101191

### Related Papers

**RDEX Framework**:
Tanis, C., Heathcote, A., Zrubka, M., & Matzke, D. (2022). A hybrid approach to dynamic cognitive psychometrics. *PsyArXiv*.

**Racing Diffusion**:
Logan, G.D., Van Zandt, T., Verbruggen, F., & Wagenmakers, E.-J. (2014). On the ability to inhibit thought and action: general and special theories of an act of control. *Psychological Review, 121*(1), 66.

**BEESTS (Ex-Gaussian Stop Process)**:
Matzke, D., Dolan, C.V., Logan, G.D., Brown, S.D., & Wagenmakers, E.-J. (2013). Bayesian parametric estimation of stop-signal reaction time distributions. *Journal of Experimental Psychology: General, 142*(4), 1047.

**Trigger Failure**:
Matzke, D., Love, J., & Heathcote, A. (2017). A Bayesian approach for estimating the probability of trigger failures in the stop-signal paradigm. *Behavior Research Methods, 49*(1), 267-281.

## Troubleshooting

### "Model takes too long to fit"
- Start with fewer subjects for testing
- Use `draws=500, tune=500` for quick diagnostics
- Consider using informative priors from pilot data

### "Divergent transitions"
- Check for extreme RTs (< 0.15s or > 2s)
- Verify SSD values are reasonable
- May need to adjust priors for your specific data

### "Parameters not recovering well"
- Ensure sufficient stop trials (paper recommends 60+)
- Check that SSDs span a reasonable range
- Verify choice accuracy increases with SSD (the key pattern)

## Support

For questions about:
- **The model**: See paper (Weigard et al., 2023)
- **Implementation**: Check docstrings in `rdex_abcd.py`
- **PyMC**: https://www.pymc.io/
- **ABCD Study**: https://abcdstudy.org/

## License

This implementation is provided for research purposes. Please cite the original paper (Weigard et al., 2023) when using this model.
