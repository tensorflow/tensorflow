# Macros for building CUDA code used with Bazel remote
# execution service.
# DO NOT EDIT: automatically generated file

def if_cuda(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with CUDA.

    Returns a select statement which evaluates to if_true if we're building
    with CUDA enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "@local_config_cuda//cuda:using_nvcc": if_true,
        "@local_config_cuda//cuda:using_clang": if_true,
        "//conditions:default": if_false
    })


def cuda_default_copts():
    """Default options for all CUDA compilations."""
    return if_cuda(["-x", "cuda", "-DGOOGLE_CUDA=1"] + ["--cuda-gpu-arch=sm_30"])


def cuda_is_configured():
    """Returns true if CUDA was enabled during the configure process."""
    return True

def if_cuda_is_configured(x):
    """Tests if the CUDA was enabled during the configure process.

    Unlike if_cuda(), this does not require that we are building with
    --config=cuda. Used to allow non-CUDA code to depend on CUDA libraries.
    """
    if cuda_is_configured():
      return x
    return []

