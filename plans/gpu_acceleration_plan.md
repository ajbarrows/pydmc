# GPU Acceleration Implementation Plan for RDEX-ABCD PyMC Models

**Date**: 2025-12-24
**Status**: READY FOR IMPLEMENTATION
**Priority**: HIGH
**Target Platform**: UVM VACC HPC Cluster (RHEL 9.4, Slurm, H100/A100/V100 GPUs)

---

## Executive Summary

This plan provides a phased approach to GPU-accelerate the hierarchical RDEX-ABCD Bayesian model implemented in PyMC. The current CPU-only implementation is computationally intensive (2 hours for 10 subjects, 24+ hours for 100 subjects). By enabling JAX backend and GPU acceleration, we can achieve **5-40x speedups**, making 100+ subject analyses tractable (under 1-2 hours).

**User Requirements**:
- Typical use case: 100+ subjects (large-scale ABCD study data)
- Preferred approach: Quick wins first, then evaluate if more optimization needed
- Precision: Comfortable with float32 on GPU (2-3x additional speedup)
- Hardware: VACC cluster with H100 (80GB), A100 (40GB), and V100 (32GB) GPUs

**Implementation Strategy**:
1. **Phase 1 (Quick Wins)**: Enable JAX + GPU with minimal code changes → 5-10x speedup in 1-2 days
2. **Phase 2 (Advanced)**: Vectorization + optimization if Phase 1 insufficient → additional 2-5x speedup

---

## Background Context

### Project Overview

`pydmc` is a Python implementation of the RDEX-ABCD cognitive model for analyzing stop-signal task data from the ABCD study. The model was converted from Stan to PyMC for easier debugging.

**Key characteristics**:
- **11 model parameters** per subject across 3 trial types (go, signal-respond, successful stop)
- **Hierarchical structure**: Group-level priors + individual-level parameters
- **Custom likelihoods**: Racing diffusion (Wald approximation) + Ex-Gaussian distributions
- **SSD-dependent drift rates**: Critical innovation accounting for stimulus replacement
- **Current dependencies**: PyMC 5.25.1, PyTensor 2.31.7, Python 3.10 (managed via Pixi)

### Current Computational Bottlenecks

Analysis of `pydmc/models.py` (1143 lines) identified these hotspots:

1. **Custom likelihood computations** (lines 840-1039):
   - 3 separate `pm.Potential` blocks with complex log-likelihood calculations
   - Go trials (lines 840-883): Wald distribution + go failure mixture
   - Signal-respond trials (lines 885-963): Racing diffusion + Ex-Gaussian CDF + mixtures
   - Successful stops (lines 965-1039): Ex-Gaussian CDF + integration approximation

2. **Ex-Gaussian CDF approximation** (lines 698-729):
   - Called for every stop trial (signal-respond + successful stop)
   - Multiple transcendental functions (exp, erf, sqrt)
   - Numerical stability clipping operations
   - Not optimized for GPU

3. **Hierarchical parameter indexing**:
   - 15 group-level hyperparameters
   - 11 individual-level parameters × N subjects
   - Array indexing on every trial: `B_go = B[self.data['go_subject_idx']]`
   - Repeated for each parameter and trial type

4. **Centered parameterization** (lines 778-807):
   - Current implementation: `B = pm.Lognormal('B', mu=mu_B, sigma=sigma_B, shape=n_subjects)`
   - Can cause poor sampling geometry in hierarchical models
   - Non-centered would improve convergence

5. **No JAX backend configured**:
   - Currently using PyTensor's default CPU backend
   - PyMC 5.25.1 + PyTensor 2.31.7 support JAX but it's not enabled
   - No GPU acceleration despite model being JAX-compatible

### VACC Cluster Specifications

**GPU Resources**:
- **H100 nodes** (2): 4× H100 80GB per node, 64-core Intel Xeon, 1TB RAM
- **A100 nodes** (2): 2× A100 40GB per node, 128-core AMD EPYC, 1TB RAM
- **V100 nodes** (10): 8× V100 32GB per node, 32-core Intel Xeon Gold

**Software Environment**:
- OS: RHEL 9.4
- Scheduler: Slurm 24.05.4
- Modules: Lmod system
- CUDA: 11.4 baseline (11.8-12.9.1 available via modules)

**Memory estimates for 100 subjects** (4 chains):
- Formula: ~(2 + 0.3 × N_subjects) GB per chain
- 100 subjects: ~32 GB total → fits on A100/H100, tight on V100

---

## PHASE 1: Quick Wins (Priority: HIGH)

**Effort**: 1-2 days
**Expected Speedup**: 5-10x
**Risk**: Low (graceful fallback to CPU)

### Objective

Enable JAX backend with GPU acceleration using minimal code changes. This phase focuses on infrastructure setup and configuration rather than algorithmic optimization.

### Implementation Steps

#### 1.1 Add JAX Dependencies

**File**: `pixi.toml`
**Location**: Lines 8-18 (dependencies section)

**Action**: Add JAX to dependencies:
```toml
[dependencies]
# ... existing dependencies ...
python = "~=3.10.0"
pymc = ">=5.25.1,<6"
numpy = ">=2.2.6,<3"
# ADD THESE:
jax = { version = ">=0.4.28,<0.5" }
jaxlib = { version = ">=0.4.28,<0.5" }
```

**Then run**:
```bash
pixi install
```

**Notes**:
- JAX from conda-forge auto-detects CUDA version from loaded modules
- No need for separate CUDA-specific JAX builds
- Compatible with CUDA 11.8-12.x available on VACC
- PyTensor 2.31.7 has full JAX backend support built-in

#### 1.2 Create GPU Configuration Module

**New File**: `pydmc/gpu_utils.py`

**Purpose**: GPU detection, configuration, and environment optimization

**Required Functions**:

```python
def detect_gpu_environment() -> Tuple[str, Optional[int]]:
    """
    Detect GPU type and memory on VACC using nvidia-smi.

    Returns
    -------
    gpu_type : str
        'H100', 'A100', 'V100', or 'CPU'
    gpu_memory_gb : int or None
        GPU memory in GB
    """
    # Implementation:
    # - Run nvidia-smi --query-gpu=name,memory.total
    # - Parse output to identify H100/A100/V100
    # - Extract memory in GB
    # - Return ('CPU', None) if no GPU found
```

```python
def configure_pytensor_gpu(device='auto', floatX='float32', verbose=True):
    """
    Configure PyTensor for GPU execution with JAX backend.

    Parameters
    ----------
    device : str
        'auto' (detect), 'cpu', or 'cuda'
    floatX : str
        'float32' (recommended for GPU) or 'float64'
    verbose : bool
        Print configuration info

    Returns
    -------
    dict : Applied configuration
    """
    # Implementation:
    # 1. Detect GPU if device='auto'
    # 2. Set pytensor.config.device = 'cuda' or 'cpu'
    # 3. Set pytensor.config.floatX
    # 4. Set pytensor.config.optimizer = 'fast_run'
    # 5. Configure JAX XLA memory settings:
    #    - H100 (80GB): XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
    #    - A100 (40GB): XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
    #    - V100 (32GB): XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
    # 6. Set XLA_PYTHON_CLIENT_PREALLOCATE=true
    # 7. Return config dict with device, floatX, gpu_type, gpu_memory
```

```python
def get_optimal_chain_config(n_chains=4):
    """
    Get optimal MCMC chain configuration for available GPU.

    Returns
    -------
    dict : Recommended chains, cores, and rationale
    """
    # Implementation logic:
    # - CPU: cores=chains for parallelization
    # - H100/A100: chains=4, cores=1 (GPU parallelizes within chain)
    # - V100: chains=2, cores=1 (memory constrained)
    # - Return dict with 'chains', 'cores', 'recommendation'
```

**Key Design Principle**: Auto-detection with graceful fallback. User doesn't need to manually configure GPU type.

#### 1.3 Modify Model Classes

**File**: `pydmc/models.py`

**Changes to `RDEXABCDModel.__init__` (around line 525)**:

```python
class RDEXABCDModel:
    def __init__(self, use_hierarchical: bool = True, use_gpu: bool = True):
        """
        Initialize RDEX-ABCD model with optional GPU acceleration.

        Parameters
        ----------
        use_hierarchical : bool
            Use hierarchical structure for multiple subjects (default: True)
        use_gpu : bool
            Enable GPU acceleration if available (default: True)
        """
        self.use_hierarchical = use_hierarchical
        self.use_gpu = use_gpu
        self.model = None
        self.trace = None
        self.data = None
        self.gpu_config = None

        # Configure GPU
        if use_gpu:
            try:
                from pydmc.gpu_utils import configure_pytensor_gpu
                self.gpu_config = configure_pytensor_gpu(
                    device='auto',
                    floatX='float32',  # Will be overridden by fit() if needed
                    verbose=False
                )
            except Exception as e:
                warnings.warn(f"GPU setup failed: {e}. Using CPU backend.")
                self.gpu_config = {'device': 'cpu', 'gpu_type': 'CPU'}
```

**Changes to `fit()` method (around line 1044)**:

```python
def fit(self, data_df: pd.DataFrame,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        cores: Optional[int] = None,
        use_float32: Optional[bool] = None,
        **kwargs) -> az.InferenceData:
    """
    Fit the RDEX-ABCD model to data.

    Parameters
    ----------
    data_df : pd.DataFrame
        Data with columns: subject, stimulus, response, rt, ssd
    draws : int
        Number of posterior samples per chain (default: 1000)
    tune : int
        Number of tuning/warmup steps (default: 1000)
    chains : int
        Number of MCMC chains (default: 4)
    cores : int, optional
        Number of cores for parallelization.
        Auto-detected: 1 for GPU (sequential chains), chains for CPU
    use_float32 : bool, optional
        Use float32 precision (faster on GPU).
        Auto-detected: True for GPU, False for CPU
    **kwargs
        Additional arguments passed to pm.sample()

    Returns
    -------
    trace : az.InferenceData
        Posterior samples and diagnostics
    """
    import pytensor

    # Auto-detect float precision
    if use_float32 is None:
        use_float32 = (self.gpu_config is not None and
                       self.gpu_config.get('device') == 'cuda')

    # Set precision
    if use_float32:
        pytensor.config.floatX = 'float32'
        print("Using float32 precision for GPU acceleration")
    else:
        pytensor.config.floatX = 'float64'
        print("Using float64 precision")

    # Auto-detect cores
    if cores is None:
        if self.use_gpu and self.gpu_config.get('device') == 'cuda':
            cores = 1  # Sequential chains on GPU
            print(f"GPU detected: {self.gpu_config.get('gpu_type')} - using cores=1")
        else:
            cores = chains  # Parallel chains on CPU
            print(f"CPU mode - using cores={cores}")

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
            cores=cores,
            **kwargs
        )

    print("✓ Sampling complete!")
    return self.trace
```

**Also modify `WaldStopSignalModel`** in same file (lines 37-479) with identical changes for consistency.

#### 1.4 Update HPC Utilities

**File**: `pydmc/utils.py`
**Function**: `setup_hpc_environment()` (currently line 13)

**Modification**:

```python
def setup_hpc_environment(temp_dir=None, use_gpu=True, verbose=True):
    """
    Setup HPC environment for pydmc with GPU support.

    Configures writable temp directory (for HPC permission issues)
    and GPU acceleration (if available).

    Parameters
    ----------
    temp_dir : str, optional
        Path to writable temp directory. Defaults to $HOME/tmp
    use_gpu : bool
        Attempt to configure GPU acceleration (default: True)
    verbose : bool
        Print configuration info (default: True)

    Returns
    -------
    dict : Configuration info (temp_dir, gpu_config)
    """
    # ... existing temp directory setup code ...

    # Add GPU configuration
    gpu_config = None
    if use_gpu:
        try:
            from pydmc.gpu_utils import configure_pytensor_gpu
            gpu_config = configure_pytensor_gpu(device='auto', verbose=verbose)
        except ImportError:
            if verbose:
                warnings.warn("gpu_utils not available, skipping GPU setup")
        except Exception as e:
            if verbose:
                warnings.warn(f"GPU setup failed: {e}")

    return {
        'temp_dir': temp_dir,
        'gpu_config': gpu_config
    }
```

#### 1.5 Create Slurm Job Templates

**New Directory**: `slurm_templates/`

Create four job templates for different GPU types and CPU fallback.

**Template 1: `slurm_templates/h100_job.sh`** (Recommended for 100+ subjects)

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=rdex_h100
#SBATCH --output=logs/rdex_%j.out
#SBATCH --error=logs/rdex_%j.err

echo "=========================================="
echo "RDEX-ABCD Model Fitting on H100"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load CUDA module (H100 requires CUDA 12+)
module load cuda/12.9.1
echo "✓ CUDA 12.9.1 loaded"

# Navigate to project directory
cd /gpfs1/home/a/j/ajbarrow/phd/projects/ABCD/pydmc

# Activate pixi environment
eval "$(pixi shell-hook)"
echo "✓ Pixi environment activated"

# Setup HPC environment
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR
mkdir -p logs
echo "✓ Temp directory: $TMPDIR"

# JAX GPU memory settings for H100 (80GB)
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
echo "✓ JAX memory settings configured (75% of 80GB)"

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Run fitting script
python << 'EOF'
from pydmc import RDEXABCDModel
from pydmc.utils import setup_hpc_environment
import pandas as pd
import time

# Setup
print("\n=== Setting up environment ===")
setup_hpc_environment(use_gpu=True)

# Load data
print("\n=== Loading data ===")
data = pd.read_csv('data/processed/abcd_stop_signal.csv')
print(f"Loaded {len(data)} trials from {data['subject'].nunique()} subjects")

# Fit model
print("\n=== Fitting RDEX-ABCD model ===")
model = RDEXABCDModel(use_hierarchical=True, use_gpu=True)

start_time = time.time()
trace = model.fit(
    data,
    chains=4,
    draws=1000,
    tune=1000,
    target_accept=0.95,
    random_seed=42
)
elapsed = time.time() - start_time

print(f"\n✓ Fitting complete in {elapsed/60:.1f} minutes")

# Save results
print("\n=== Saving results ===")
trace.to_netcdf('results/trace_h100.nc')
model.summary()

print(f"\n{'='*50}")
print(f"Job completed successfully!")
print(f"Total time: {elapsed/60:.1f} minutes")
print(f"{'='*50}")
EOF

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
```

**Template 2: `slurm_templates/a100_job.sh`**

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=rdex_a100
#SBATCH --output=logs/rdex_%j.out
#SBATCH --error=logs/rdex_%j.err

echo "RDEX-ABCD Model Fitting on A100"
module load cuda/12.2.2
cd /gpfs1/home/a/j/ajbarrow/phd/projects/ABCD/pydmc
eval "$(pixi shell-hook)"

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR logs

# A100 (40GB) memory settings
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Same Python script as H100, just change output filename:
# trace.to_netcdf('results/trace_a100.nc')
python fit_model.py  # Or inline script
```

**Template 3: `slurm_templates/v100_job.sh`**

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=rdex_v100
#SBATCH --output=logs/rdex_%j.out
#SBATCH --error=logs/rdex_%j.err

echo "RDEX-ABCD Model Fitting on V100"
module load cuda/11.8.0
cd /gpfs1/home/a/j/ajbarrow/phd/projects/ABCD/pydmc
eval "$(pixi shell-hook)"

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR logs

# V100 (32GB) memory settings - more conservative
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# For 100+ subjects, use 2 chains instead of 4 in Python:
# trace = model.fit(data, chains=2, ...)
python fit_model.py
```

**Template 4: `slurm_templates/cpu_fallback.sh`**

```bash
#!/bin/bash
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=rdex_cpu
#SBATCH --output=logs/rdex_%j.out
#SBATCH --error=logs/rdex_%j.err

echo "RDEX-ABCD Model Fitting on CPU"
cd /gpfs1/home/a/j/ajbarrow/phd/projects/ABCD/pydmc
eval "$(pixi shell-hook)"

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR logs

# No GPU modules, no XLA settings

# CPU fitting with use_gpu=False, cores=32
python << 'EOF'
from pydmc import RDEXABCDModel
import pandas as pd

data = pd.read_csv('data/processed/abcd_stop_signal.csv')
model = RDEXABCDModel(use_hierarchical=True, use_gpu=False)
trace = model.fit(data, chains=4, cores=32, draws=1000, tune=1000)
trace.to_netcdf('results/trace_cpu.nc')
EOF
```

#### 1.6 Validation and Testing

**Quick Validation Script** (run interactively on compute node):

```python
#!/usr/bin/env python
"""Quick validation that GPU acceleration is working."""

from pydmc import RDEXABCDModel, simulate_from_config
from pydmc.utils import setup_hpc_environment
import time
import numpy as np

print("=" * 60)
print("GPU ACCELERATION VALIDATION")
print("=" * 60)

# Setup
print("\n1. Setting up environment...")
config = setup_hpc_environment(use_gpu=True, verbose=True)

# Check JAX
print("\n2. Checking JAX GPU detection...")
try:
    import jax
    devices = jax.devices()
    print(f"   JAX devices: {devices}")
    if any('gpu' in str(d).lower() for d in devices):
        print("   ✓ GPU detected by JAX")
    else:
        print("   ⚠ No GPU detected - will use CPU")
except Exception as e:
    print(f"   ✗ JAX error: {e}")

# Simulate small dataset
print("\n3. Simulating test data (5 subjects)...")
data, true_params = simulate_from_config(
    'examples/configs/default_params.yaml',
    n_subjects=5,
    seed=42
)
print(f"   Simulated {len(data)} trials")

# CPU baseline
print("\n4. Fitting on CPU (baseline)...")
model_cpu = RDEXABCDModel(use_gpu=False)
start = time.time()
trace_cpu = model_cpu.fit(data, draws=100, tune=100, chains=2, random_seed=42)
cpu_time = time.time() - start
print(f"   CPU time: {cpu_time:.1f} seconds")

# GPU comparison
print("\n5. Fitting on GPU...")
model_gpu = RDEXABCDModel(use_gpu=True)
start = time.time()
trace_gpu = model_gpu.fit(data, draws=100, tune=100, chains=2, random_seed=42)
gpu_time = time.time() - start
print(f"   GPU time: {gpu_time:.1f} seconds")

# Results
speedup = cpu_time / gpu_time
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"CPU time:  {cpu_time:.1f}s")
print(f"GPU time:  {gpu_time:.1f}s")
print(f"Speedup:   {speedup:.1f}x")

# Validate convergence
print("\nConvergence diagnostics:")
rhat_cpu = np.max(trace_cpu.rhat().to_array().values)
rhat_gpu = np.max(trace_gpu.rhat().to_array().values)
print(f"CPU R̂ max: {rhat_cpu:.3f} (should be < 1.01)")
print(f"GPU R̂ max: {rhat_gpu:.3f} (should be < 1.01)")

# Validate parameter estimates match
mu_v_plus_cpu = trace_cpu.posterior['mu_v_plus'].mean().values
mu_v_plus_gpu = trace_gpu.posterior['mu_v_plus'].mean().values
rel_error = abs(mu_v_plus_cpu - mu_v_plus_gpu) / mu_v_plus_cpu
print(f"\nParameter agreement (mu_v_plus):")
print(f"CPU: {mu_v_plus_cpu:.3f}")
print(f"GPU: {mu_v_plus_gpu:.3f}")
print(f"Relative error: {rel_error:.1%} (should be < 5%)")

# Summary
print("\n" + "=" * 60)
if speedup > 2 and rhat_gpu < 1.01 and rel_error < 0.05:
    print("✓ VALIDATION PASSED")
    print(f"  - Speedup achieved: {speedup:.1f}x")
    print(f"  - Convergence: R̂ < 1.01")
    print(f"  - Results match CPU")
else:
    print("⚠ VALIDATION ISSUES")
    if speedup < 2:
        print(f"  - Low speedup: {speedup:.1f}x (expected > 2x)")
    if rhat_gpu >= 1.01:
        print(f"  - Poor convergence: R̂ = {rhat_gpu:.3f}")
    if rel_error >= 0.05:
        print(f"  - Parameter mismatch: {rel_error:.1%} error")
print("=" * 60)
```

**Save as**: `scripts/validate_gpu.py` and run with:
```bash
# On a GPU node
srun --partition=gpu --gpus-per-node=h100:1 --mem=32G --time=00:30:00 \
  --pty bash -c "cd /gpfs1/home/a/j/ajbarrow/phd/projects/ABCD/pydmc && \
  module load cuda/12.9.1 && \
  eval \"\$(pixi shell-hook)\" && \
  python scripts/validate_gpu.py"
```

**Validation Checklist**:

- [ ] JAX installed: `pixi list | grep jax`
- [ ] JAX detects GPU: `python -c "import jax; print(jax.devices())"`
- [ ] PyTensor can use CUDA: `python -c "import pytensor; pytensor.config.device='cuda'; print(pytensor.config.device)"`
- [ ] Model fits without errors on GPU
- [ ] Speedup > 3x on small test (5 subjects, 100 draws)
- [ ] Convergence: R̂ < 1.01 for all parameters
- [ ] ESS_bulk > 100 (for short test)
- [ ] No excessive divergences (< 1%)
- [ ] Parameter estimates match CPU (< 5% relative error)
- [ ] Can submit and run Slurm job successfully

---

## PHASE 2: Advanced Optimizations (Priority: MEDIUM)

**Effort**: 1 week
**Expected Additional Speedup**: 2-5x (on top of Phase 1)
**Risk**: Medium (requires careful validation)

**When to implement**: Only if Phase 1 performance is insufficient for your 100+ subject use cases. Evaluate Phase 1 results first.

### Objective

Optimize model code for better GPU utilization through vectorization and improved parameterization.

### 2.1 Vectorize Custom Likelihoods

**Problem**: Current implementation indexes parameters separately for each trial type, resulting in many small array operations that don't fully utilize GPU parallelism.

**Example of current approach** (lines 845-849):
```python
B_go = B[self.data['go_subject_idx']]
t0_go = t0[self.data['go_subject_idx']]
v_plus_go = v_plus[self.data['go_subject_idx']]
v_minus_go = v_minus[self.data['go_subject_idx']]
pgf_go = pgf[self.data['go_subject_idx']]
```

**Optimized approach**:
```python
# Stack all parameters into single array
go_params = pm.math.stack([B, t0, v_plus, v_minus, pgf], axis=1)
# Single indexing operation
go_params_indexed = go_params[self.data['go_subject_idx']]
# Unpack
B_go, t0_go, v_plus_go, v_minus_go, pgf_go = [
    go_params_indexed[:, i] for i in range(5)
]
```

**Files to modify**: `pydmc/models.py`

**Sections to vectorize**:
1. **Go trials likelihood** (lines 840-883)
2. **Signal-respond trials likelihood** (lines 885-963)
3. **Successful stop trials likelihood** (lines 965-1039)

**Expected impact**: 2-3x speedup by reducing memory operations and improving GPU cache utilization.

### 2.2 Non-Centered Parameterization

**Problem**: Centered parameterization can cause poor sampling geometry in hierarchical models, leading to slow mixing and divergences.

**Current approach** (line 778):
```python
B = pm.Lognormal('B', mu=mu_B, sigma=sigma_B, shape=n_subjects)
```

**Non-centered approach**:
```python
# Sample from standard normal
B_offset = pm.Normal('B_offset', mu=0, sigma=1, shape=n_subjects)

# Transform to actual parameter
B = pm.Deterministic('B',
    pm.math.exp(pm.math.log(mu_B) + sigma_B * B_offset))
```

**Files to modify**: `pydmc/models.py` (lines 778-807)

**Parameters to convert** (all 11 individual-level parameters):
- `B`, `t0`, `v_plus`, `v_minus`, `v0`, `g` (go process)
- `stop_mu`, `stop_sigma`, `stop_tau` (stop process)
- `pgf`, `ptf` (failure probabilities)

**Benefits**:
- Better sampling efficiency (20-30% faster convergence)
- Fewer divergences
- More robust for extreme group-level variances

**Expected impact**: 1.3-1.5x speedup through better sampling efficiency.

### 2.3 Optimize Ex-Gaussian CDF (Optional, Advanced)

**Current implementation** (lines 698-729) is a custom approximation with multiple transcendental functions. Potential optimizations:

**Option A**: Lookup table with interpolation (fastest, less accurate)
**Option B**: JAX-optimized analytical approximation (balanced)
**Option C**: Custom CUDA kernel (most complex, maximum speed)

**Recommended**: Option B - rewrite approximation using JAX primitives that JIT-compile efficiently.

**Expected impact**: 1.5-2x additional speedup (diminishing returns).

**Only implement if**: Profiling shows Ex-Gaussian CDF is still a bottleneck after 2.1 and 2.2.

---

## Performance Expectations

### Phase 1 Performance (JAX + GPU + float32)

| N Subjects | CPU (32 cores) | V100 | A100 | H100 | Speedup (H100) |
|------------|----------------|------|------|------|----------------|
| 10         | 2 hours        | 25 min | 20 min | 15 min | **8x** |
| 50         | 10 hours       | 90 min | 70 min | 50 min | **12x** |
| 100        | 24 hours       | 3 hours | 2 hours | 90 min | **16x** |
| 200        | 60 hours       | 7 hours | 5 hours | 3.5 hours | **17x** |

**Assumptions**: 1000 draws, 1000 tune, 4 chains (2 chains for V100 with 100+ subjects)

### Phase 1 + Phase 2 Performance (Full Optimization)

| N Subjects | V100 | A100 | H100 | Total Speedup vs CPU |
|------------|------|------|------|----------------------|
| 10         | 10 min | 8 min | 5 min | **24x** |
| 50         | 45 min | 30 min | 20 min | **30x** |
| 100        | 75 min | 50 min | 35 min | **40x** |
| 200        | 3 hours | 2 hours | 1.5 hours | **40x** |

### Memory Constraints

**Formula**: ~(2 + 0.3 × N_subjects) GB per chain for 4 chains

| N Subjects | Total Memory (4 chains) | V100 (32GB) | A100 (40GB) | H100 (80GB) |
|------------|-------------------------|-------------|-------------|-------------|
| 10         | 5 GB                    | ✓ (4 chains) | ✓ (4 chains) | ✓ (4 chains) |
| 50         | 17 GB                   | ✓ (2-4 chains) | ✓ (4 chains) | ✓ (4 chains) |
| 100        | 32 GB                   | ⚠ (2 chains) | ✓ (4 chains) | ✓ (4 chains) |
| 200        | 62 GB                   | ✗ | ⚠ (2 chains) | ✓ (4 chains) |

**Mitigation strategies**:
- Use 2 chains instead of 4
- Use float32 (saves ~40% memory)
- Split data across multiple jobs
- Use H100 nodes for large datasets

---

## Implementation Timeline

### Week 1: Infrastructure & Testing

**Days 1-2**: Setup
- [ ] Add JAX dependencies to pixi.toml
- [ ] Run `pixi install` and verify JAX installation
- [ ] Create `pydmc/gpu_utils.py` with detection and configuration functions
- [ ] Test GPU detection: `python -c "from pydmc.gpu_utils import detect_gpu_environment; print(detect_gpu_environment())"`

**Days 3-4**: Model modifications
- [ ] Modify `RDEXABCDModel.__init__` to accept `use_gpu` parameter
- [ ] Modify `RDEXABCDModel.fit()` to handle float32 and auto-detect cores
- [ ] Modify `WaldStopSignalModel` similarly for consistency
- [ ] Update `setup_hpc_environment()` in `pydmc/utils.py`

**Days 5-6**: Slurm templates
- [ ] Create `slurm_templates/` directory
- [ ] Write H100, A100, V100, and CPU job templates
- [ ] Test job submission (dry run)
- [ ] Create validation script `scripts/validate_gpu.py`

**Day 7**: Validation
- [ ] Run validation script on H100 node (or available GPU)
- [ ] Verify 3-5x speedup on small test
- [ ] Check convergence diagnostics
- [ ] Benchmark on 10 subjects with realistic settings (1000 draws, 1000 tune)

### Week 2: Production Testing

**Days 8-10**: Scale testing
- [ ] Fit 10, 20, 50 subjects on GPU
- [ ] Measure actual speedups
- [ ] Monitor memory usage
- [ ] Check for divergences or convergence issues

**Days 11-13**: Real data validation
- [ ] Load ABCD stop-signal data subset
- [ ] Fit model on real data (10-20 subjects)
- [ ] Validate posterior predictive checks
- [ ] Compare to any existing results

**Day 14**: Decision point
- [ ] Evaluate Phase 1 performance
- [ ] Decide if Phase 2 optimizations needed
- [ ] Document findings and benchmarks

### Week 3+ (Optional): Phase 2 Implementation

Only proceed if Phase 1 insufficient for 100+ subjects:

**Days 15-17**: Vectorization
- [ ] Vectorize go trials likelihood
- [ ] Vectorize signal-respond likelihood
- [ ] Vectorize successful stop likelihood
- [ ] Validate results match original implementation

**Days 18-20**: Non-centered parameterization
- [ ] Convert all 11 parameters to non-centered
- [ ] Test convergence on 10 subjects
- [ ] Compare divergence rates to centered version

**Day 21**: Phase 2 validation
- [ ] Full benchmark on 50-100 subjects
- [ ] Measure additional speedup
- [ ] Final validation and documentation

---

## Critical Files Summary

### Must Create (Phase 1):

1. **`pydmc/gpu_utils.py`** (NEW) - ~200 lines
   - `detect_gpu_environment()`
   - `configure_pytensor_gpu()`
   - `get_optimal_chain_config()`

2. **`slurm_templates/h100_job.sh`** (NEW) - Slurm job for H100
3. **`slurm_templates/a100_job.sh`** (NEW) - Slurm job for A100
4. **`slurm_templates/v100_job.sh`** (NEW) - Slurm job for V100
5. **`slurm_templates/cpu_fallback.sh`** (NEW) - CPU-only fallback

6. **`scripts/validate_gpu.py`** (NEW) - ~100 lines validation script

### Must Modify (Phase 1):

7. **`pixi.toml`** (lines 8-18)
   - Add jax and jaxlib dependencies

8. **`pydmc/models.py`** (1143 lines total)
   - `RDEXABCDModel.__init__` (line ~525): Add `use_gpu` parameter
   - `RDEXABCDModel.fit` (line ~1044): Add `use_float32`, auto-detect `cores`
   - `WaldStopSignalModel.__init__` (line ~37): Same changes
   - `WaldStopSignalModel.fit` (line ~225): Same changes

9. **`pydmc/utils.py`** (67 lines total)
   - `setup_hpc_environment` (line ~13): Add GPU configuration call

### Optional (Phase 2):

10. **`pydmc/models.py`** (additional changes)
    - Lines 840-883: Vectorize go trials likelihood
    - Lines 885-963: Vectorize signal-respond likelihood
    - Lines 965-1039: Vectorize successful stop likelihood
    - Lines 778-807: Non-centered parameterization

11. **`tests/test_gpu.py`** (NEW) - Unit tests
12. **`benchmarks/benchmark_gpu.py`** (NEW) - Systematic benchmarking

---

## Validation Strategy

### Unit Tests

After Phase 1 implementation, validate:

1. **GPU detection works**: `detect_gpu_environment()` returns correct GPU type
2. **PyTensor configuration**: Device set to 'cuda' when GPU available
3. **Model initialization**: `use_gpu=True` doesn't raise errors
4. **Fit completes**: Model fits without crashing on GPU
5. **Results match CPU**: Parameter estimates within 5% of CPU version
6. **Convergence**: R̂ < 1.01, ESS_bulk > 400 for typical run
7. **No excess divergences**: < 1% divergent transitions
8. **Speedup achieved**: > 3x on 10 subjects, > 5x on 50+ subjects

### Parameter Recovery Test

```python
from pydmc import RDEXABCDModel, simulate_from_config
import numpy as np

# Simulate with known parameters
data, true_params = simulate_from_config(
    'examples/configs/default_params.yaml',
    n_subjects=10,
    seed=42
)

# Fit on GPU
model = RDEXABCDModel(use_gpu=True)
trace = model.fit(data, draws=1000, tune=1000, chains=4)

# Check recovery
for param in ['mu_v_plus', 'mu_v_minus', 'mu_B', 'mu_t0']:
    est = trace.posterior[param].mean().values
    true = true_params[param.replace('mu_', '')].mean()
    rel_error = abs(est - true) / true

    print(f"{param}: true={true:.3f}, est={est:.3f}, error={rel_error:.1%}")
    assert rel_error < 0.10, f"{param} not recovered (error={rel_error:.1%})"

print("✓ Parameter recovery validated")
```

### Posterior Predictive Checks

```python
# Generate posterior predictive samples
with model.model:
    ppc = pm.sample_posterior_predictive(trace)

# Check key patterns
# 1. Choice accuracy increases with SSD
# 2. Signal-respond RT > Go RT
# 3. Inhibition function shape
# 4. RT distributions match data

# Implement specific checks based on RDEX-ABCD paper
```

### Benchmark Script

```python
import time
import pandas as pd
from pydmc import RDEXABCDModel, simulate_from_config

results = []

for n_subj in [10, 20, 50, 100]:
    print(f"\nBenchmarking {n_subj} subjects...")

    data, _ = simulate_from_config('default_params.yaml', n_subjects=n_subj)

    # CPU
    model_cpu = RDEXABCDModel(use_gpu=False)
    start = time.time()
    trace_cpu = model_cpu.fit(data, draws=500, tune=500, chains=2)
    cpu_time = time.time() - start

    # GPU
    model_gpu = RDEXABCDModel(use_gpu=True)
    start = time.time()
    trace_gpu = model_gpu.fit(data, draws=500, tune=500, chains=2)
    gpu_time = time.time() - start

    results.append({
        'n_subjects': n_subj,
        'cpu_time_min': cpu_time / 60,
        'gpu_time_min': gpu_time / 60,
        'speedup': cpu_time / gpu_time,
        'rhat_max_cpu': trace_cpu.rhat().to_array().max().values,
        'rhat_max_gpu': trace_gpu.rhat().to_array().max().values,
    })

df = pd.DataFrame(results)
print(df)
df.to_csv('benchmark_results.csv')
```

---

## Success Criteria

### Phase 1 Success:
- ✓ JAX installed and GPU detected
- ✓ Model fits without errors on GPU
- ✓ 5-10x speedup vs CPU on 10+ subjects
- ✓ Convergence: R̂ < 1.01 for all parameters
- ✓ ESS_bulk > 400 per parameter (for 1000 draws)
- ✓ < 1% divergent transitions
- ✓ Parameter recovery: < 10% error on known parameters
- ✓ Results reproducible across runs (with same seed)
- ✓ Can fit 100 subjects in < 2 hours on A100/H100

### Phase 2 Success (if implemented):
- ✓ Additional 2-4x speedup (15-40x total vs CPU)
- ✓ Can fit 100 subjects in < 1 hour on H100
- ✓ No degradation in convergence
- ✓ No increase in divergences
- ✓ Parameter estimates unchanged (< 2% difference from Phase 1)

---

## Troubleshooting Guide

### Issue: JAX doesn't detect GPU

**Check**:
```bash
module load cuda/12.9.1
python -c "import jax; print(jax.devices())"
```

**If shows CPU only**:
- Verify CUDA module loaded: `module list`
- Check nvidia-smi works: `nvidia-smi`
- Check JAX installation: `pixi list | grep jax`
- Try explicit CUDA path: `export LD_LIBRARY_PATH=/path/to/cuda/lib64:$LD_LIBRARY_PATH`

### Issue: Out of memory on GPU

**Solutions**:
1. Reduce chains: Use 2 instead of 4
2. Reduce batch size: Fit fewer subjects per job
3. Use float32: Saves ~40% memory
4. Use larger GPU: H100 instead of V100
5. Reduce draws: Start with 500/500 instead of 1000/1000

**Check memory usage**:
```bash
watch -n 1 nvidia-smi
```

### Issue: Divergent transitions

**Not GPU-specific**, but common in hierarchical models:

**Solutions**:
1. Increase `target_accept`: 0.95 or 0.99 (slower but more accurate)
2. Use Phase 2 non-centered parameterization
3. Check for extreme RTs in data (< 0.15s or > 2s)
4. Verify SSD values reasonable
5. Check priors aren't too restrictive

### Issue: Slower than CPU

**Possible causes**:
1. Dataset too small (< 5 subjects): GPU overhead > benefit
2. Not using float32: Use `use_float32=True`
3. Multiple chains on GPU: Should use `cores=1`
4. CUDA not loaded: Check `module list`
5. Memory swapping: Reduce memory usage

**Profile**:
```python
import jax
with jax.profiler.trace("/tmp/jax-trace"):
    trace = model.fit(data, draws=100, tune=100)
# Then analyze trace with TensorBoard
```

### Issue: Results don't match CPU

**If relative error > 5%**:

**Possible causes**:
1. Different random seeds: Set `random_seed=42` explicitly
2. Float32 precision: Normal up to ~2-3% difference
3. Insufficient draws: Use more samples
4. Poor convergence: Check R̂ and ESS

**Validate**:
```python
# Run both with same seed
trace_cpu = model_cpu.fit(data, random_seed=42, ...)
trace_gpu = model_gpu.fit(data, random_seed=42, ...)

# Compare posteriors
import arviz as az
az.plot_forest([trace_cpu, trace_gpu], model_names=['CPU', 'GPU'])
```

---

## Additional Resources

### Documentation
- **PyMC GPU guide**: https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/posterior_predictive.html
- **JAX documentation**: https://jax.readthedocs.io/
- **PyTensor backends**: https://pytensor.readthedocs.io/en/latest/

### VACC Resources
- **Cluster specs**: https://www.uvm.edu/vacc/cluster-specs
- **Slurm guide**: https://www.uvm.edu/vacc/kb/slurm
- **GPU usage**: https://www.uvm.edu/vacc/kb/gpu

### Model Documentation
- **RDEX-ABCD paper**: Weigard et al. (2023), Dev Cogn Neurosci
- **Project README**: `/gpfs1/home/a/j/ajbarrow/phd/projects/ABCD/pydmc/README.md`
- **CLAUDE.md**: `/gpfs1/home/a/j/ajbarrow/phd/projects/ABCD/pydmc/CLAUDE.md`

---

## Next Actions

1. **Review this plan** and confirm it aligns with your needs
2. **Start with Phase 1, Step 1.1**: Add JAX dependencies to pixi.toml
3. **Work through Phase 1 sequentially**: Each step builds on the previous
4. **Validate early and often**: Test GPU detection before modifying model code
5. **Benchmark at each step**: Measure actual speedups to guide decisions
6. **Decide on Phase 2**: After Phase 1 results, evaluate if additional optimization needed

---

## Notes

- This plan was created based on analysis of the current codebase as of 2024-12-24
- The existing `setup_hpc_environment()` utility already handles temp directory permissions
- PyMC 5.25.1 + PyTensor 2.31.7 have full JAX support built-in
- For 200+ subjects, may need multi-GPU or batching strategies (not covered here)
- The plan assumes reasonable access to H100/A100 nodes; adjust timeline if limited GPU availability

---

**End of Plan**