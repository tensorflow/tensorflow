load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Macros for building ROCm code.
def if_rocm(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with ROCm.

    Returns a select statement which evaluates to if_true if we're building
    with ROCm enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "@local_config_rocm//rocm:using_hipcc": if_true,
        "//conditions:default": if_false,
    })

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
    return if_true if %{rocm_is_configured} else if_false

def is_rocm_configured():
    """
    Returns True if ROCm is configured. False otherwise.
    """
    return %{rocm_is_configured}

def if_rocm_hipblaslt(x):
    """ 
    hipBlasLt is always available: kept for compatibility with Tensorflow.
    """
    return select({"//conditions:default": x})

def get_rbe_amdgpu_pool(is_single_gpu = False):
    return "%{single_gpu_rbe_pool}" if is_single_gpu else "%{multi_gpu_rbe_pool}"

def rocm_lib_import(name, interface_library, data, deps):
    cc_import(
        name = name + "_interface",
        shared_library = interface_library,
        visibility = ["//visibility:private"],
    )
    cc_library(
        name = name + "_libs",
        data = data,
        deps = deps,
        visibility = ["//visibility:private"],
    )
    cc_library(
        name = name,
        deps = [
            ":{}_interface".format(name),
            ":{}_libs".format(name),
            ":rocm_headers_includes",
            ":rocm_rpath",
        ],
        visibility = ["//visibility:public"],
    )
