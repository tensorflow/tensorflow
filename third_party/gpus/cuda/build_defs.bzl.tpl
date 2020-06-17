# Macros for building CUDA code.
def if_cuda(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with CUDA.

    Returns a select statement which evaluates to if_true if we're building
    with CUDA enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "@local_config_cuda//cuda:using_nvcc": if_true,
        "@local_config_cuda//cuda:using_clang": if_true,
        "//conditions:default": if_false,
    })

def if_cuda_clang(if_true, if_false = []):
   """Shorthand for select()'ing on wheteher we're building with cuda-clang.

    Returns a select statement which evaluates to if_true if we're building
    with cuda-clang.  Otherwise, the select statement evaluates to if_false.

   """
   return select({
       "@local_config_cuda//cuda:using_clang": if_true,
       "//conditions:default": if_false
   })

def if_cuda_clang_opt(if_true, if_false = []):
   """Shorthand for select()'ing on wheteher we're building with cuda-clang
   in opt mode.

    Returns a select statement which evaluates to if_true if we're building
    with cuda-clang in opt mode. Otherwise, the select statement evaluates to
    if_false.

   """
   return select({
       "@local_config_cuda//cuda:using_clang_opt": if_true,
       "//conditions:default": if_false
   })

def cuda_default_copts():
    """Default options for all CUDA compilations."""
    return if_cuda([
        "-x", "cuda",
        "-DGOOGLE_CUDA=1",
        "-Xcuda-fatbinary=--compress-all",
        "--no-cuda-include-ptx=all"
    ] + %{cuda_extra_copts}) + if_cuda_clang_opt(
        # Some important CUDA optimizations are only enabled at O3.
        ["-O3"]
    )

def cuda_is_configured():
    """Returns true if CUDA was enabled during the configure process."""
    return %{cuda_is_configured}

def cuda_gpu_architectures():
    """Returns a list of supported GPU architectures."""
    return %{cuda_gpu_architectures}

def if_cuda_is_configured(x):
    """Tests if the CUDA was enabled during the configure process.

    Unlike if_cuda(), this does not require that we are building with
    --config=cuda. Used to allow non-CUDA code to depend on CUDA libraries.
    """
    if cuda_is_configured():
      return select({"//conditions:default": x})
    return select({"//conditions:default": []})

def cuda_header_library(
        name,
        hdrs,
        include_prefix = None,
        strip_include_prefix = None,
        deps = [],
        **kwargs):
    """Generates a cc_library containing both virtual and system include paths.

    Generates both a header-only target with virtual includes plus the full
    target without virtual includes. This works around the fact that bazel can't
    mix 'includes' and 'include_prefix' in the same target."""

    native.cc_library(
        name = name + "_virtual",
        hdrs = hdrs,
        include_prefix = include_prefix,
        strip_include_prefix = strip_include_prefix,
        deps = deps,
        visibility = ["//visibility:private"],
    )

    native.cc_library(
        name = name,
        textual_hdrs = hdrs,
        deps = deps + [":%s_virtual" % name],
        **kwargs
    )

def cuda_library(copts = [], **kwargs):
    """Wrapper over cc_library which adds default CUDA options."""
    native.cc_library(copts = cuda_default_copts() + copts, **kwargs)
