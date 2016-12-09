# Macros for building CUDA code.
def cuda_path_flags():
    """Stub for compatibility with internal build."""
    return []

def if_cuda(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with CUDA.

    Returns a select statement which evaluates to if_true if we're building
    with CUDA enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "@local_config_cuda//cuda:using_nvcc": if_true,
        "//conditions:default": if_false
    })

def cuda_is_configured():
    """Returns true if CUDA was enabled during the configure process."""
    return %{cuda_is_configured}

def if_cuda_is_configured(x):
    """Tests if the CUDA was enabled during the configure process.

    Unlike if_cuda(), this does not require that we are building with
    --config=cuda. Used to allow non-CUDA code to depend on CUDA libraries.
    """
    if cuda_is_configured():
      return x
    return []
