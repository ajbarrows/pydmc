# Debugging Stan Errors in pydmc

## Quick Fix: See the Actual Error

If you're getting cryptic errors, enable console output:

```python
from pydmc import WaldStopSignalModel

model = WaldStopSignalModel(use_hierarchical=True, centered_parameterization=True)

# Show the actual Stan error messages!
fit = model.fit(
    data,
    chains=2,
    iter=500,
    warmup=250,
    show_console=True  # <- This shows what's really happening
)
```

## Common Errors and Solutions

### 1. "Exception: ...  is not a valid value"
**Cause**: Data contains invalid values (NaN, negative RTs, etc.)

**Fix**:
```python
import numpy as np

# Check for issues
print("NaN RTs:", data['rt'].isna().sum())
print("Negative RTs:", (data['rt'] < 0).sum())
print("Zero RTs:", (data['rt'] == 0).sum())

# Clean data
data = data[data['rt'] > 0.1].copy()  # Remove very short RTs
data = data[data['rt'] < 5.0].copy()  # Remove very long RTs
```

### 2. "Gradient evaluation...  Informational Message"
**Cause**: Parameter values causing numerical issues during sampling

**Possible fixes**:
```python
# Option 1: Use centered parameterization (more stable)
model = WaldStopSignalModel(
    use_hierarchical=True,
    centered_parameterization=True  # <- More stable
)

# Option 2: Increase adapt_delta
fit = model.fit(data, adapt_delta=0.95)  # Default is 0.8

# Option 3: Reduce initial step size
fit = model.fit(data, step_size=0.01)
```

### 3. "Initialization failed" or "Rejecting initial value"
**Cause**: Model can't find valid starting values

**Fix**:
```python
# Check your data ranges
print("RT range:", data['rt'].min(), "-", data['rt'].max())
print("Response types:", data['response'].value_counts())

# Make sure you have variability
print("Subjects:", data['subject'].nunique())
print("Trials per subject:", data.groupby('subject').size())
```

### 4. High Rhat (> 1.01) or divergent transitions
**Cause**: Sampler not converging

**Fixes** (try in order):
```python
# 1. More warmup and iterations
fit = model.fit(data, chains=4, iter=3000, warmup=2000)

# 2. Higher adapt_delta
fit = model.fit(data, adapt_delta=0.95)

# 3. Use centered parameterization
model = WaldStopSignalModel(
    use_hierarchical=True,
    centered_parameterization=True
)

# 4. Fewer subjects initially (for testing)
test_data = data[data['subject'].isin(data['subject'].unique()[:3])]
fit = model.fit(test_data, ...)
```

### 5. "Operation not permitted" (HPC systems)
**Cause**: Can't write to `/tmp`

**Fix**: This should now be automatic, but if not:
```python
from pydmc import setup_hpc_environment
setup_hpc_environment()

# Then run your model
model.fit(data, ...)
```

### 6. Very slow sampling
**Normal behavior**:
- First chain is slowest (compilation)
- Each iteration shows progress
- ~1-10 seconds per iteration is normal

**If unusually slow**:
```python
# Start with fewer iterations for testing
fit = model.fit(data, chains=2, iter=200, warmup=100)

# Use individual model first (faster)
model = WaldStopSignalModel(use_hierarchical=False)

# Reduce data for initial testing
test_data = data.sample(n=100)
```

## Diagnostic Workflow

```python
from pydmc import WaldStopSignalModel, print_environment_info
import pandas as pd

# 1. Check environment
print_environment_info()

# 2. Load and check data
data = pd.read_csv('your_data.csv')
print(f"Data shape: {data.shape}")
print(f"Subjects: {data['subject'].nunique()}")
print(f"RT range: {data['rt'].min():.3f} - {data['rt'].max():.3f}")
print(f"Response distribution:\n{data['response'].value_counts()}")

# 3. Start simple
print("\n=== Testing with individual model ===")
model = WaldStopSignalModel(use_hierarchical=False)

try:
    # Very short run for testing
    fit = model.fit(
        data,
        chains=1,
        iter=100,
        warmup=50,
        show_console=True  # See what's happening
    )
    print("✓ Individual model works!")

except Exception as e:
    print(f"✗ Error: {e}")
    print("\nTrying with subset of data...")
    test_data = data.sample(n=50)
    fit = model.fit(test_data, chains=1, iter=100, warmup=50, show_console=True)

# 4. Move to hierarchical if individual works
print("\n=== Testing hierarchical model ===")
model = WaldStopSignalModel(
    use_hierarchical=True,
    centered_parameterization=True  # More stable
)

fit = model.fit(
    data,
    chains=2,
    iter=500,
    warmup=250,
    show_console=True
)

print("✓ Success!")
model.summary()
```

## Advanced Debugging

### View Raw Stan Output
```python
# After getting an error, check the output files
if model.fit_result is not None:
    print(model.fit_result.runset._retcodes)
    print(model.fit_result.runset.stdout)
```

### Check Diagnostics
```python
# After successful fit
estimates = model.get_parameter_estimates()

# Check Rhat values
for param, values in estimates.items():
    if 'rhat' in str(values).lower():
        print(f"{param}: Rhat = ...")

# Use ArviZ for detailed diagnostics
import arviz as az
az_data = az.from_cmdstanpy(model.fit_result)
az.plot_trace(az_data)
```

### Test with Synthetic Data
```python
# Generate simple synthetic data
import numpy as np

np.random.seed(42)
synthetic = []
for i in range(100):
    synthetic.append({
        'subject': 'S01',
        'stimulus': np.random.choice([0, 1]),
        'response': np.random.choice([1, 2]),
        'rt': np.random.uniform(0.4, 0.8),
        'ssd': np.nan
    })

test_data = pd.DataFrame(synthetic)

# If this works, your model is fine - check your actual data
model.fit(test_data, chains=1, iter=100, warmup=50)
```

## Getting Help

If none of these work:

1. Run `notebooks/debug-sampling.ipynb`
2. Check `TROUBLESHOOTING.md`
3. Enable `show_console=True` and save the output
4. Check the Stan forums: https://discourse.mc-stan.org/

When asking for help, provide:
- Full error message with `show_console=True`
- Data summary (shape, ranges, distributions)
- Model configuration
- System info from `print_environment_info()`
