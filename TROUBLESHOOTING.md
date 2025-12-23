# Troubleshooting Guide for pydmc

## Common Stan Sampling Errors

### 1. "Error during sampling" - General

This error can have several causes. Try these steps in order:

#### A. Check Your Data Format
```python
import pandas as pd

# Your data should look like this:
data = pd.DataFrame({
    'subject': ['S01', 'S01', 'S01', ...],
    'stimulus': [0, 1, 0, ...],        # 0=left, 1=right
    'response': [1, 2, 0, ...],        # 0=no response, 1=left, 2=right
    'rt': [0.45, 0.52, np.nan, ...],   # In SECONDS, NaN for no-response trials
    'ssd': [np.nan, np.nan, 0.25, ...]  # NaN for go trials, value for stop trials
})
```

#### B. Run the Debug Notebook
```bash
jupyter notebook notebooks/debug-sampling.ipynb
```

This will systematically check:
- Stan backend installation
- Data validity
- Model compilation
- Sampling with minimal data

#### C. Start with Individual-Level Model
```python
# Simpler model is easier to debug
model = WaldStopSignalModel(use_hierarchical=False)

# Use short runs for testing
fit = model.fit(data, chains=1, iter=100, warmup=50)
```

### 2. Backend Issues

#### CmdStanPy Not Found
```bash
pip install cmdstanpy
python -m cmdstanpy.install_cmdstan
```

#### PyStan Not Found
```bash
pip install pystan==2.19.1.1
```

### 3. Data Issues

#### Very Short or Long RTs
Filter your data to reasonable ranges:
```python
# RTs should be in seconds, typically 0.2-2.0 for most tasks
data = data[(data['rt'] >= 0.15) & (data['rt'] <= 5.0) | (data['rt'].isna())]
```

#### Insufficient Trials
- Minimum: ~50 trials per subject for individual models
- Recommended: 100+ trials per subject for hierarchical models

#### No Response Variability
If all responses are identical, the model cannot estimate parameters:
```python
# Check response distribution
print(data['response'].value_counts())

# Should have at least 2 different response types
```

### 4. Model Convergence Issues

#### Divergent Transitions
Try these in order:
1. Increase warmup: `warmup=1500` or `warmup=2000`
2. Increase adapt_delta: `model.fit(data, adapt_delta=0.95)`
3. Use more iterations: `iter=3000` or `iter=4000`

#### Low ESS (Effective Sample Size)
- Increase iterations: `iter=4000, warmup=2000`
- Check for parameter identifiability issues

#### High Rhat values (> 1.01)
- Run more chains: `chains=6` or `chains=8`
- Increase iterations
- Check trace plots for convergence issues

### 5. Memory Issues

For large datasets or hierarchical models:
```python
# Reduce chains initially
fit = model.fit(data, chains=2, iter=1000)

# Don't include samples when saving
model.save_results('results.json', include_samples=False)
```

### 6. Slow Sampling

#### Normal for First Run
- Model compilation takes 30-60 seconds
- First sampling can take 5-30 minutes depending on data size

#### Speed Tips
- Reduce iterations for testing: `iter=500, warmup=250`
- Use fewer chains: `chains=2`
- Use CmdStanPy backend (faster than PyStan)
- Reduce data size for initial testing

### 7. Data Validation Errors

The model now validates your data before fitting. Common warnings:

#### "Very short RTs detected"
```python
# Filter out unrealistically fast responses
data = data[data['rt'] >= 0.15]
```

#### "Low correct response rate"
Check your stimulus-response coding:
- stimulus=0 should match response=1 (left)
- stimulus=1 should match response=2 (right)

#### "Some subjects have < 30 trials"
```python
# Remove subjects with insufficient data
trial_counts = data.groupby('subject').size()
valid_subjects = trial_counts[trial_counts >= 30].index
data = data[data['subject'].isin(valid_subjects)]
```

## Getting Help

### 1. Check the Full Error Message
Always look at the complete traceback, not just the first line.

### 2. Verify Your Stan Installation
```python
from pydmc import WaldStopSignalModel
model = WaldStopSignalModel()
print(f"Backend: {model.backend.backend_name}")
```

### 3. Test with Synthetic Data
Use the example from `01-basic-example.ipynb` which generates valid synthetic data.

### 4. Examine Stan Output
If sampling starts but fails, Stan may print diagnostic information. Look for:
- "Rejecting initial value" - data or priors may be incompatible
- "Exception" - check data types and array sizes
- "Gradient evaluation" - numerical issues with parameters

## Quick Diagnostic Checklist

- [ ] Stan backend installed (CmdStanPy or PyStan)
- [ ] Data has required columns: subject, stimulus, response, rt
- [ ] Response codes are 0, 1, or 2
- [ ] Stimulus codes are 0 or 1
- [ ] RTs are in seconds (not milliseconds)
- [ ] At least 50 trials per subject
- [ ] Multiple response types present
- [ ] No NaN values in required columns (except RT for no-response trials)
- [ ] Tried with individual-level model first
- [ ] Tried with reduced iterations for testing

## Still Having Issues?

1. Run `notebooks/debug-sampling.ipynb`
2. Check your data format carefully
3. Try the synthetic data example first
4. Open an issue with:
   - Full error traceback
   - Data summary (shape, columns, first few rows)
   - Code you're running
