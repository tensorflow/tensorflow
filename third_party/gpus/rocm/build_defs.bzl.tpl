# Macros for building ROCm code.
def if_rocm(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with ROCm.

    Returns a select statement which evaluates to if_true if we're building
    with ROCm enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "@local_config_rocm//rocm:using_hipcc": if_true,
        "//conditions:default": if_false
    })

def select_threshold(value, above_or_eq, threshold, below):
    return below if value < threshold else above_or_eq

def rocm_default_copts():
    """Default options for all ROCm compilations."""
    return if_rocm(["-x", "rocm"] + %{rocm_extra_copts})

def rocm_copts(opts = []):
    """Gets the appropriate set of copts for (maybe) ROCm compilation.

      If we're doing ROCm compilation, returns copts for our particular ROCm
      compiler.  If we're not doing ROCm compilation, returns an empty list.

      """
    return rocm_default_copts() + select({
        "//conditions:default": [],
        "@local_config_rocm//rocm:using_hipcc": ([
            "",
        ]),
    }) + if_rocm_is_configured(opts)

def rocm_gpu_architectures():
    """Returns a list of supported GPU architectures."""
    return %{rocm_gpu_architectures}

def rocm_version_number():
    """Returns a list of supported GPU architectures."""
    return %{rocm_version_number}

def if_gpu_is_configured(if_true, if_false = []):
    """Tests if ROCm or CUDA or SYCL was enabled during the configure process."""
    return select({"//conditions:default": %{gpu_is_configured}})

def if_cuda_or_rocm(if_true, if_false = []):
    """Tests if ROCm or CUDA was enabled during the configure process.

    Unlike if_rocm() or if_cuda(), this does not require that we are building
    with --config=rocm or --config=cuda, respectively. Used to allow non-GPU
    code to depend on ROCm or CUDA libraries.

    """
    return select({"//conditions:default": %{cuda_or_rocm}})

def if_rocm_is_configured(if_true, if_false = []):
    """Tests if the ROCm was enabled during the configure process.

    Unlike if_rocm(), this does not require that we are building with
    --config=rocm. Used to allow non-ROCm code to depend on ROCm libraries.
    """
    if %{rocm_is_configured}:
      return select({"//conditions:default": if_true})
    return select({"//conditions:default": if_false})

def is_rocm_configured():
    """
    Returns True if ROCm is configured. False otherwise.
    """
    return %{rocm_is_configured}

def rocm_hipblaslt():
    return %{rocm_is_configured} and %{rocm_hipblaslt}

def if_rocm_hipblaslt(x):
    if %{rocm_is_configured} and (%{rocm_hipblaslt} == "True"):
      return select({"//conditions:default": x})
    return select({"//conditions:default": []})

def rocm_library(copts = [], **kwargs):
    """Wrapper over cc_library which adds default ROCm options."""
    native.cc_library(copts = rocm_default_copts() + copts, **kwargs)
