"""Provides build configuration for TSL"""

load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
)
load(
    "@local_config_tensorrt//:build_defs.bzl",
    "if_tensorrt",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm",
)
load(
    "//third_party/mkl:build_defs.bzl",
    "if_enable_mkl",
    "if_mkl",
)
load(
    "//third_party/mkl_dnn:build_defs.bzl",
    "if_mkldnn_aarch64_acl",
    "if_mkldnn_aarch64_acl_openmp",
    "if_mkldnn_openmp",
)
load(
    "//third_party/compute_library:build_defs.bzl",
    "if_enable_acl",
)

def clean_dep(target):
    """Returns string to 'target' in @org_tensorflow repository.

    Use this function when referring to targets in the @org_tensorflow
    repository from macros that may be called from external repositories.
    """

    # A repo-relative label is resolved relative to the file in which the
    # Label() call appears, i.e. @org_tensorflow.
    return str(Label(target))

def if_oss(oss_value, google_value = []):
    """Returns one of the arguments based on the non-configurable build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    return oss_value  # copybara:comment_replace return google_value

def if_google(google_value, oss_value = []):
    """Returns one of the arguments based on the non-configurable build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    return oss_value  # copybara:comment_replace return google_value

def if_libtpu(if_true, if_false = []):
    """Shorthand for select()ing whether to build backend support for TPUs when building libtpu.so"""
    return select({
        # copybara:uncomment_begin(different config setting in OSS)
        # "//tools/cc_target_os:gce": if_true,
        # copybara:uncomment_end_and_comment_begin
        clean_dep("//tensorflow:with_tpu_support"): if_true,
        # copybara:comment_end
        "//conditions:default": if_false,
    })

def if_windows(a, otherwise = []):
    return select({
        clean_dep("//tensorflow/tsl:windows"): a,
        "//conditions:default": otherwise,
    })

def if_not_windows(a):
    return select({
        clean_dep("//tensorflow/tsl:windows"): [],
        "//conditions:default": a,
    })

def if_nvcc(a):
    return select({
        "@local_config_cuda//cuda:using_nvcc": a,
        "//conditions:default": [],
    })

def if_xla_available(if_true, if_false = []):
    return select({
        clean_dep("//tensorflow/tsl:with_xla_support"): if_true,
        "//conditions:default": if_false,
    })

def if_android_arm(a):
    return select({
        clean_dep("//tensorflow/tsl:android_arm"): a,
        "//conditions:default": [],
    })

def if_linux_x86_64(a):
    return select({
        clean_dep("//tensorflow/tsl:linux_x86_64"): a,
        "//conditions:default": [],
    })

def if_ios_x86_64(a):
    return select({
        clean_dep("//tensorflow/tsl:ios_x86_64"): a,
        "//conditions:default": [],
    })

def if_no_default_logger(a):
    return select({
        clean_dep("//tensorflow/tsl:no_default_logger"): a,
        "//conditions:default": [],
    })

def get_win_copts(is_external = False):
    WINDOWS_COPTS = [
        "/DPLATFORM_WINDOWS",
        "/DEIGEN_HAS_C99_MATH",
        "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
        "/DEIGEN_AVOID_STL_ARRAY",
        "/Iexternal/gemmlowp",
        "/wd4018",  # -Wno-sign-compare
        # Bazel's CROSSTOOL currently pass /EHsc to enable exception by
        # default. We can't pass /EHs-c- to disable exception, otherwise
        # we will get a waterfall of flag conflict warnings. Wait for
        # Bazel to fix this.
        # "/D_HAS_EXCEPTIONS=0",
        # "/EHs-c-",
        "/wd4577",
        "/DNOGDI",
        # Also see build:windows lines in tensorflow/opensource_only/.bazelrc
        # where we set some other options globally.
    ]
    if is_external:
        return WINDOWS_COPTS + ["/UTF_COMPILE_LIBRARY"]
    else:
        return WINDOWS_COPTS + ["/DTF_COMPILE_LIBRARY"]

def tf_copts(
        android_optimization_level_override = "-O2",
        is_external = False,
        allow_exceptions = False):
    # For compatibility reasons, android_optimization_level_override
    # is currently only being set for Android.
    # To clear this value, and allow the CROSSTOOL default
    # to be used, pass android_optimization_level_override=None
    android_copts = [
        "-DTF_LEAN_BINARY",
        "-Wno-narrowing",
        "-fomit-frame-pointer",
    ]
    if android_optimization_level_override:
        android_copts.append(android_optimization_level_override)
    return (
        if_not_windows([
            "-DEIGEN_AVOID_STL_ARRAY",
            "-Iexternal/gemmlowp",
            "-Wno-sign-compare",
            "-ftemplate-depth=900",
        ]) +
        (if_not_windows(["-fno-exceptions"]) if not allow_exceptions else []) +
        if_cuda(["-DGOOGLE_CUDA=1"]) +
        if_nvcc(["-DTENSORFLOW_USE_NVCC=1"]) +
        if_libtpu(["-DLIBTPU_ON_GCE"], []) +
        if_xla_available(["-DTENSORFLOW_USE_XLA=1"]) +
        if_tensorrt(["-DGOOGLE_TENSORRT=1"]) +
        if_rocm(["-DTENSORFLOW_USE_ROCM=1"]) +
        # Compile in oneDNN based ops when building for x86 platforms
        if_mkl(["-DINTEL_MKL"]) +
        # Enable additional ops (e.g., ops with non-NHWC data layout) and
        # optimizations for Intel builds using oneDNN if configured
        if_enable_mkl(["-DENABLE_MKL"]) +
        if_mkldnn_openmp(["-DENABLE_ONEDNN_OPENMP"]) +
        if_mkldnn_aarch64_acl(["-DDNNL_AARCH64_USE_ACL=1"]) +
        if_mkldnn_aarch64_acl_openmp(["-DENABLE_ONEDNN_OPENMP"]) +
        if_enable_acl(["-DXLA_CPU_USE_ACL=1", "-fexceptions"]) +
        if_android_arm(["-mfpu=neon"]) +
        if_linux_x86_64(["-msse3"]) +
        if_ios_x86_64(["-msse4.1"]) +
        if_no_default_logger(["-DNO_DEFAULT_LOGGER"]) +
        select({
            clean_dep("//tensorflow/tsl:framework_shared_object"): [],
            "//conditions:default": ["-DTENSORFLOW_MONOLITHIC_BUILD"],
        }) +
        select({
            clean_dep("//tensorflow/tsl:android"): android_copts,
            clean_dep("//tensorflow/tsl:emscripten"): [],
            clean_dep("//tensorflow/tsl:macos"): [],
            clean_dep("//tensorflow/tsl:windows"): get_win_copts(is_external),
            clean_dep("//tensorflow/tsl:ios"): [],
            clean_dep("//tensorflow/tsl:no_lgpl_deps"): ["-D__TENSORFLOW_NO_LGPL_DEPS__", "-pthread"],
            "//conditions:default": ["-pthread"],
        })
    )

def tf_openmp_copts():
    # We assume when compiling on Linux gcc/clang will be used and MSVC on Windows
    return select({
        # copybara:uncomment_begin
        # "//third_party/mkl:build_with_mkl_lnx_openmp": ["-fopenmp"],
        # "//third_party/mkl:build_with_mkl_windows_openmp": ["/openmp"],
        # copybara:uncomment_end_and_comment_begin
        "@org_tensorflow//third_party/mkl:build_with_mkl_lnx_openmp": ["-fopenmp"],
        "@org_tensorflow//third_party/mkl:build_with_mkl_windows_openmp": ["/openmp:llvm"],
        # copybara:comment_end
        "//conditions:default": [],
    })
