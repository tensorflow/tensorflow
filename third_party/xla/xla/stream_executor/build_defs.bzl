"""Configurations for StreamExecutor builds"""

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    _if_cuda_or_rocm = "if_cuda_or_rocm",
    _if_gpu_is_configured = "if_gpu_is_configured",
)
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

def stream_executor_friends():
    return ["//..."]

def stream_executor_gpu_friends():
    return ["//..."]

def stream_executor_internal():
    return ["//..."]

def tf_additional_cuda_platform_deps():
    return []

def tf_additional_cudnn_plugin_copts():
    return ["-DNV_CUDNN_DISABLE_EXCEPTION"]

# Returns whether any GPU backend is configured.
def if_gpu_is_configured(if_true, if_false = []):
    return _if_gpu_is_configured(if_true, if_false)

def if_cuda_or_rocm(if_true, if_false = []):
    return _if_cuda_or_rocm(if_true, if_false)

# nvlink is not available via the pip wheels, disable it since it will create
# unnecessary dependency
def tf_additional_gpu_compilation_copts():
    return ["-DTF_DISABLE_NVLINK_BY_DEFAULT"]

def gpu_only_cc_library(name, tags = [], **kwargs):
    """A library that only gets compiled when GPU is configured, otherwise it's an empty target.

    Args:
      name: Name of the target
      tags: Tags being applied to the implementation target
      **kwargs: Accepts all arguments that a `cc_library` would also accept
    """
    if not native.package_name().startswith("xla/stream_executor"):
        fail("gpu_only_cc_library may only be used in `xla/stream_executor/...`.")

    cc_library(
        name = "%s_non_gpu" % name,
        tags = ["manual"],
    )
    cc_library(
        name = "%s_gpu_only" % name,
        tags = tags + ["manual"],
        **kwargs
    )
    native.alias(
        name = name,
        actual = if_gpu_is_configured(":%s_gpu_only" % name, ":%s_non_gpu" % name),
        visibility = kwargs.get("visibility"),
        compatible_with = kwargs.get("compatible_with"),
        restricted_to = kwargs.get("restricted_to"),
        target_compatible_with = kwargs.get("target_compatible_with"),
    )

def stream_executor_build_defs_bzl_deps():
    return []
