"""Configurations for StreamExecutor builds"""

load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_library",
    "if_cuda_is_configured",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
    "rocm_default_copts",
    _if_cuda_or_rocm = "if_cuda_or_rocm",
    _if_gpu_is_configured = "if_gpu_is_configured",
)
load(
    "@local_tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)

def stream_executor_friends():
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

def gpu_only_cc_library(
        name,
        cuda_srcs = [],
        cuda_deps = [],
        rocm_srcs = [],
        rocm_deps = [],
        impl_hdrs = [],
        **kwargs):
    """A library target that helps managing backend specific code.

       * The target is an empty target when no GPU-specific backend is enabled.
       * It exposes `hdrs` and compiles `srcs` when any GPU-specific backend is enabled.
       * It compiles `cuda_srcs` when CUDA is enabled.
         `cuda_srcs` may depend on `cuda_deps` and may include `impl_hdrs`.
       * It compiles `rocm_srcs` when ROCm is enabled.
         `rocm_srcs` may depend on `rocm_deps` and may include `impl_hdrs`.

       The purpose of this target is to offer an easy way to handle backend specific code
       paths which doesn't require preprocessor branches (`#ifdef GOOGLE_CUDA` etc.).

       You can use `impl_hdrs` to declare a function or a type that is only visible to
       source files in this target. For example you declare a function and implement it
       in different ways for ROCm and CUDA. Then you can call it from non-backend-specific
       code in `srcs`.

       Everything declared in `impl_hdrs` is private to the target and cannot be exposed
       to consumers.

       Note that the backend specific targets cannot include any header from `hdrs`. This is
       by design. It avoid cyclic dependencies between the backend-specific and the backend
       agnostic code and will helps us refactor things further in the future. If you need
       to share code between the backend-specific and -agnostic targets, create a separate
       library target and depend on it via `deps` and `cuda/rocm_deps`.

       Dependency structure:
           <name> --x---> <name>_gpu_only --x---> <name>_cuda_only
                     `--> <name>_non_gpu     `--> <name>_rocm_only
       `-->` means "depends on". "x"s are decision points where only one path is taken.

    Args:
      name: Name of the target
      cuda_srcs: A list of CUDA only source files
      cuda_deps: A list of CUDA only dependencies
      rocm_srcs: A list of ROCm only source files
      rocm_deps: A list of ROCm only dependencies
      impl_hdrs: A list of private headers that can only be included from source files in `srcs`,
                 `cuda_srcs`, and `rocm_srcs`.
      **kwargs: Accepts all arguments that a `cc_library` would also accept
    """
    if not native.package_name().startswith("xla/stream_executor"):
        fail("gpu_only_cc_library may only be used in `xla/stream_executor/...`.")

    tags = kwargs.pop("tags", default = [])
    impl_target_tags = tags + ["manual", "alt_dep=:%s" % name, "avoid_dep"]
    deps = kwargs.pop("deps", default = [])
    srcs = kwargs.pop("srcs", default = [])
    hdrs = kwargs.pop("hdrs", default = [])

    cc_library(
        name = "%s_non_gpu" % name,
        tags = impl_target_tags,
    )
    cc_library(
        name = "%s_gpu_only" % name,
        srcs = srcs + impl_hdrs,
        tags = impl_target_tags,
        deps = deps +
               if_cuda_is_configured([":%s_cuda_only" % name]) +
               if_rocm_is_configured([":%s_rocm_only" % name]),
        hdrs = hdrs,
        **kwargs
    )

    common_lib_attributes = {}
    if "visibility" in kwargs:
        common_lib_attributes["visibility"] = kwargs["visibility"]
    if "compatible_with" in kwargs:
        common_lib_attributes["compatible_with"] = kwargs["compatible_with"]
    if "restricted_to" in kwargs:
        common_lib_attributes["restricted_to"] = kwargs["restricted_to"]
    if "target_compatible_with" in kwargs:
        common_lib_attributes["target_compatible_with"] = kwargs["target_compatible_with"]

    # Backend specific code needs to be kept in separate targets for build_cleaner to work.
    # build_cleaner can't deal with `srcs = if_cuda_is_configured([...])`.
    cuda_library(
        name = "%s_cuda_only" % name,
        srcs = cuda_srcs + impl_hdrs,
        tags = impl_target_tags,
        deps = cuda_deps + [":%s_public_headers" % name],
        **common_lib_attributes
    )
    cuda_library(
        name = "%s_rocm_only" % name,
        srcs = rocm_srcs + impl_hdrs,
        tags = impl_target_tags,
        local_defines = rocm_default_copts(),
        deps = rocm_deps + [":%s_public_headers" % name],
        **common_lib_attributes
    )

    cc_library(
        name = "%s_public_headers" % name,
        hdrs = hdrs,
        deps = deps,
        tags = impl_target_tags,
        # We disable the layering check and header parsing to avoid any build system complaints
        # about this internal target. Layering and self contained headers will still be enforced by
        # the `<name>_gpu_only` target which has the same hdrs and deps.
        features = ["-layering_check", "-parse_headers"],
    )

    native.alias(
        name = name,
        actual = if_gpu_is_configured("%s_gpu_only" % name, "%s_non_gpu" % name),
        **common_lib_attributes
    )

def cuda_only_cc_library(name, tags = [], **kwargs):
    """A library that only gets compiled when CUDA is configured, otherwise it's an empty target.

    Args:
      name: Name of the target
      tags: Tags being applied to the implementation target
      **kwargs: Accepts all arguments that a `cc_library` would also accept
    """
    if not native.package_name().startswith("xla/stream_executor"):
        fail("cuda_only_cc_library may only be used in `xla/stream_executor/...`.")

    cc_library(
        name = "%s_non_cuda" % name,
        tags = ["manual"],
    )
    cc_library(
        name = "%s_cuda_only" % name,
        tags = tags + ["manual"],
        **kwargs
    )
    native.alias(
        name = name,
        actual = if_cuda_is_configured(":%s_cuda_only" % name, ":%s_non_cuda" % name),
        visibility = kwargs.get("visibility"),
        compatible_with = kwargs.get("compatible_with"),
        restricted_to = kwargs.get("restricted_to"),
        target_compatible_with = kwargs.get("target_compatible_with"),
    )
