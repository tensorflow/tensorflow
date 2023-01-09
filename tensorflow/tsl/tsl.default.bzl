"""Default (OSS) build versions of TSL general-purpose build extensions."""

load(
    "//tensorflow/tsl:tsl.bzl",
    "clean_dep",
    "two_gpu_tags",
    _filegroup = "filegroup",
    _get_compatible_with_portable = "get_compatible_with_portable",
    _if_not_mobile_or_arm_or_lgpl_restricted = "if_not_mobile_or_arm_or_lgpl_restricted",
    _internal_hlo_deps = "internal_hlo_deps",
    _tsl_grpc_cc_dependencies = "tsl_grpc_cc_dependencies",
    _tsl_pybind_extension = "tsl_pybind_extension",
)
load(
    "//tensorflow/tsl/platform:build_config.bzl",
    "tsl_cc_test",
)
load(
    "//tensorflow/tsl/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
)

get_compatible_with_portable = _get_compatible_with_portable
filegroup = _filegroup
if_not_mobile_or_arm_or_lgpl_restricted = _if_not_mobile_or_arm_or_lgpl_restricted
internal_hlo_deps = _internal_hlo_deps
tsl_grpc_cc_dependencies = _tsl_grpc_cc_dependencies
tsl_pybind_extension = _tsl_pybind_extension

def get_compatible_with_cloud():
    return []

def tsl_gpu_cc_test(
        name,
        srcs = [],
        deps = [],
        tags = [],
        data = [],
        size = "medium",
        linkstatic = 0,
        args = [],
        linkopts = [],
        **kwargs):
    """Create tests for cpu, gpu and optionally 2gpu

    Args:
      name: unique name for this test target.
      srcs: list of C and C++ files that are processed to create the binary target.
      deps: list of other libraries to be linked in to the binary target.
      tags: useful for categorizing the tests
      data: files needed by this rule at runtime.
      size: classification of how much time/resources the test requires.
      linkstatic: link the binary in static mode.
      args: command line arguments that Bazel passes to the target.
      linkopts: add these flags to the C++ linker command.
      **kwargs: Extra arguments to the rule.
    """
    targets = []
    tsl_cc_test(
        name = name + "_cpu",
        size = size,
        srcs = srcs,
        args = args,
        data = data,
        copts = if_cuda(["-DNV_CUDNN_DISABLE_EXCEPTION"]),
        linkopts = linkopts,
        linkstatic = linkstatic,
        tags = tags,
        deps = deps,
        **kwargs
    )
    targets.append(name + "_cpu")
    tsl_cc_test(
        name = name + "_gpu",
        size = size,
        srcs = srcs,
        args = args,
        data = data,
        copts = if_cuda(["-DNV_CUDNN_DISABLE_EXCEPTION"]),
        linkopts = linkopts,
        linkstatic = select({
            # TODO(allenl): Remove Mac static linking when Bazel 0.6 is out.
            clean_dep("//tensorflow/tsl:macos"): 1,
            "@local_config_cuda//cuda:using_nvcc": 1,
            "@local_config_cuda//cuda:using_clang": 1,
            "//conditions:default": 0,
        }),
        tags = tags + tf_gpu_tests_tags(),
        deps = deps,
        **kwargs
    )
    targets.append(name + "_gpu")
    if "multi_gpu" in tags or "multi_and_single_gpu" in tags:
        cleaned_tags = tags + two_gpu_tags
        if "requires-gpu-nvidia" in cleaned_tags:
            cleaned_tags.remove("requires-gpu-nvidia")
        tsl_cc_test(
            name = name + "_2gpu",
            size = size,
            srcs = srcs,
            args = args,
            data = data,
            linkopts = linkopts,
            linkstatic = select({
                # TODO(allenl): Remove Mac static linking when Bazel 0.6 is out.
                clean_dep("//tensorflow/tsl:macos"): 1,
                "@local_config_cuda//cuda:using_nvcc": 1,
                "@local_config_cuda//cuda:using_clang": 1,
                "//conditions:default": 0,
            }),
            tags = cleaned_tags,
            deps = deps,
            **kwargs
        )
        targets.append(name + "_2gpu")

    native.test_suite(name = name, tests = targets, tags = tags)
