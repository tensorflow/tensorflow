# Macros for building CUDA code.
def if_cuda(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with CUDA.

    Returns a select statement which evaluates to if_true if we're building
    with CUDA enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "@local_config_cuda//:is_cuda_enabled": if_true,
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

def cuda_compiler(if_cuda_clang, if_nvcc, neither = []):
    """Shorthand for select()'ing on wheteher we're building with cuda-clang or nvcc.

     Returns a select statement which evaluates to if_cuda_clang if we're building
     with cuda-clang, if_nvcc if we're building with NVCC.
     Otherwise, the select statement evaluates to neither.

    """
    if %{cuda_is_configured}:
        return select({
            "@local_config_cuda//cuda:using_clang": if_cuda_clang,
            "@local_config_cuda//:is_cuda_compiler_nvcc": if_nvcc,
            "//conditions:default": neither
        })
    else:
        return select({
            "//conditions:default": neither
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
    ] + %{cuda_extra_copts}) + if_cuda_clang_opt(
        # Some important CUDA optimizations are only enabled at O3.
        ["-O3"]
    ) + cuda_compiler(
        if_cuda_clang = [ "-Xcuda-fatbinary", "--compress-all"],
        if_nvcc = [
            "-Xcuda-fatbinary=--compress-all",
            # Ensure that NVCC matches clang's constexpr behavior.
            "-nvcc_options=-expt-relaxed-constexpr"
        ]
    )

def cuda_gpu_architectures():
    """Returns a list of supported GPU architectures."""
    return %{cuda_gpu_architectures}

def if_cuda_is_configured(x, no_cuda = []):
    """Tests if the CUDA was enabled during the configure process.

    Unlike if_cuda(), this does not require that we are building with
    --config=cuda. Used to allow non-CUDA code to depend on CUDA libraries.
    """
    if %{cuda_is_configured}:
      return select({"//conditions:default": x})
    return select({"//conditions:default": no_cuda})

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

def cuda_cc_test(copts = [], **kwargs):
    """Wrapper over cc_test which adds default CUDA options."""
    native.cc_test(copts = copts + if_cuda(["-DGOOGLE_CUDA=1"]), **kwargs)

EnableCudaInfo = provider()

def _enable_cuda_flag_impl(ctx):
    value = ctx.build_setting_value
    if ctx.attr.enable_override:
        print(
            "\n\033[1;33mWarning:\033[0m '--define=using_cuda_nvcc' will be " +
            "unsupported soon. Use '--@local_config_cuda//:enable_cuda' " +
            "instead."
        )
        value = True
    return EnableCudaInfo(value = value)

enable_cuda_flag = rule(
    implementation = _enable_cuda_flag_impl,
    build_setting = config.bool(flag = True),
    attrs = {"enable_override": attr.bool()},
)
