# Hierarchical Model Options in pydmc

## Overview

The `WaldStopSignalModel` supports two parameterizations for hierarchical models:

1. **Non-Centered Parameterization** (default) - Better with few subjects or weak data
2. **Centered Parameterization** - More stable with many subjects

## When to Use Each

### Non-Centered (Default)
```python
model = WaldStopSignalModel(use_hierarchical=True, centered_parameterization=False)
```

**Use when:**
- Few subjects (< 5)
- Weak data (few trials per subject)
- You want faster sampling
- Initial exploration

**Characteristics:**
- Faster per iteration
- Can have "funnel" problems with many subjects
- Works well when group-level variation is large

### Centered Parameterization
```python
model = WaldStopSignalModel(use_hierarchical=True, centered_parameterization=True)
```

**Use when:**
- Many subjects (≥ 5)
- Rich data (many trials per subject)
- Non-centered version has convergence issues
- Final analysis with full dataset

**Characteristics:**
- More stable with many subjects
- Slower per iteration
- Better when group-level variation is small
- Simpler parameter structure

## Symptoms of Wrong Parameterization

### Need Centered if you see:
- Runtime errors with many subjects
- Very high Rhat values (> 1.05)
- Divergent transitions
- Trace plots showing "funnels"
- Low effective sample size

### Can use Non-Centered if:
- Fast sampling
- Good Rhat values (< 1.01)
- High effective sample sizes
- Clean trace plots

## Example Workflow

```python
import pandas as pd
from pydmc import WaldStopSignalModel

# Load your data
data = pd.read_csv('stop_signal_data.csv')

# Try non-centered first (faster)
print("Trying non-centered parameterization...")
model_nc = WaldStopSignalModel(use_hierarchical=True, centered_parameterization=False)

try:
    fit_nc = model_nc.fit(data, chains=2, iter=500, warmup=250)
    model_nc.summary()

    # Check diagnostics
    # If good, use this model

except RuntimeError as e:
    print(f"Non-centered failed: {e}")
    print("\\nSwitching to centered parameterization...")

    # Use centered instead
    model_c = WaldStopSignalModel(use_hierarchical=True, centered_parameterization=True)
    fit_c = model_c.fit(data, chains=2, iter=500, warmup=250)
    model_c.summary()
```

## Technical Details

### Non-Centered Parameterization
- Parameters: `mu_params`, `sigma_params`, `params_raw[s]`
- Transform: `params[s] = mu + sigma * params_raw[s]`
- Separates location from scale
- Better geometry when data is weak

### Centered Parameterization
- Parameters: Direct subject parameters (`B[s]`, `t0[s]`, etc.)
- Prior: `B[s] ~ lognormal(log(mu_B), sigma_B)`
- More intuitive parameter interpretation
- Better geometry when data is strong

## Prior Differences

### Non-Centered
- Priors on hyperparameters (mu, sigma)
- Individual parameters are transformations
- More flexible but can be unstable

### Centered
- Direct priors on group means
- Direct priors on group SDs (more restrictive)
- More stable with many subjects
- Example:
  ```stan
  mu_B ~ lognormal(0, 1)
  sigma_B ~ normal(0, 0.5)  // Half-normal via constraint
  B[s] ~ lognormal(log(mu_B), sigma_B)
  ```

## Recommendations

1. **Start with non-centered** for initial exploration
2. **Switch to centered** if you encounter:
   - Runtime errors during sampling
   - Poor convergence (high Rhat)
   - Many divergent transitions
3. **Always check diagnostics**:
   - Trace plots
   - Rhat < 1.01
   - ESS > 400 per chain
4. **Increase iterations** if needed:
   - chains=4, iter=2000, warmup=1000 for final analysis

## Further Reading

- Betancourt & Girolami (2015): "Hamiltonian Monte Carlo for Hierarchical Models"
- Stan User's Guide: "Reparameterization" section
- Papaspiliopoulos et al. (2007): "A general framework for the parametrization of hierarchical models"
