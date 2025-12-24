"""
Utility functions for pydmc.

This module provides helper functions for setting up the environment,
especially for HPC systems.
"""

import os
import tempfile
import warnings


def setup_hpc_environment(temp_dir=None):
    """
    Set up environment for HPC systems.

    This function configures temporary directories to avoid permission issues
    common on HPC systems where /tmp may not be writable or may be restricted.

    Parameters
    ----------
    temp_dir : str, optional
        Custom temporary directory path. If None, uses ~/tmp

    Returns
    -------
    str
        Path to the temporary directory being used

    Examples
    --------
    >>> from pydmc.utils import setup_hpc_environment
    >>> setup_hpc_environment()
    'Using temporary directory: /home/user/tmp'

    >>> # Or specify custom location
    >>> setup_hpc_environment('/scratch/user/tmp')
    'Using temporary directory: /scratch/user/tmp'
    """
    if temp_dir is None:
        temp_dir = os.path.expanduser('~/tmp')

    # Create directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)

    # Set environment variables
    os.environ['TMPDIR'] = temp_dir
    os.environ['TEMP'] = temp_dir
    os.environ['TMP'] = temp_dir

    # Set Python's tempfile module
    tempfile.tempdir = temp_dir

    # Verify it works
    try:
        test_file = os.path.join(temp_dir, '.pydmc_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"Using temporary directory: {temp_dir}")
        return temp_dir
    except Exception as e:
        warnings.warn(
            f"Could not write to temporary directory {temp_dir}: {e}. "
            "You may encounter permission errors during model fitting."
        )
        return temp_dir


def check_environment():
    """
    Check if the environment is properly configured.

    Returns
    -------
    dict
        Dictionary with environment information

    Examples
    --------
    >>> from pydmc.utils import check_environment
    >>> info = check_environment()
    >>> print(info['temp_dir'])
    '/home/user/tmp'
    """
    info = {
        'temp_dir': tempfile.gettempdir(),
        'home_dir': os.path.expanduser('~'),
        'cwd': os.getcwd(),
    }

    # Check Stan backends
    try:
        import cmdstanpy
        info['cmdstanpy_version'] = cmdstanpy.__version__
        try:
            info['cmdstan_path'] = cmdstanpy.cmdstan_path()
        except:
            info['cmdstan_path'] = 'Not installed'
    except ImportError:
        info['cmdstanpy_version'] = 'Not installed'

    try:
        import pystan
        info['pystan_version'] = pystan.__version__
    except ImportError:
        info['pystan_version'] = 'Not installed'

    # Check temp directory writability
    try:
        test_file = os.path.join(info['temp_dir'], '.pydmc_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        info['temp_writable'] = True
    except:
        info['temp_writable'] = False

    return info


def print_environment_info():
    """
    Print detailed environment information.

    Examples
    --------
    >>> from pydmc.utils import print_environment_info
    >>> print_environment_info()
    Environment Information:
    ========================
    Temporary directory: /home/user/tmp
    ...
    """
    info = check_environment()

    print("Environment Information:")
    print("=" * 50)
    print(f"Temporary directory: {info['temp_dir']}")
    print(f"  Writable: {'Yes' if info['temp_writable'] else 'No'}")
    print(f"Home directory: {info['home_dir']}")
    print(f"Working directory: {info['cwd']}")
    print()
    print("Stan Backends:")
    print(f"  CmdStanPy: {info.get('cmdstanpy_version', 'Not installed')}")
    if 'cmdstan_path' in info:
        print(f"    Path: {info['cmdstan_path']}")
    print(f"  PyStan: {info.get('pystan_version', 'Not installed')}")
    print()

    # Recommendations
    if not info['temp_writable']:
        print("WARNING: Temporary directory is not writable!")
        print("  Run: from pydmc.utils import setup_hpc_environment")
        print("       setup_hpc_environment()")
        print()

    if info.get('cmdstanpy_version') == 'Not installed' and \
       info.get('pystan_version') == 'Not installed':
        print("WARNING: No Stan backend installed!")
        print("  Install CmdStanPy: pip install cmdstanpy")
        print("  Then run: python -m cmdstanpy.install_cmdstan")
        print()
