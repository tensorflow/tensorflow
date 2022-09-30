load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    licenses = ["notice"],
)

exports_files([
    "macros.h",
    "subgraph.h",
])

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//visibility:private",
    ],
)

cc_library(
    name = "subgraph",
    srcs = [
        "subgraph.cc",
    ],
    hdrs = [
        "subgraph.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "//tensorflow/lite/core:__subpackages__",
        "//tensorflow/lite/kernels:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:graph_info",
        "//tensorflow/lite:interpreter_options_header",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:macros",
        "//tensorflow/lite:memory_planner",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/c:common_internal",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/profiling:root_profiler",
        "//tensorflow/lite/schema:schema_fbs",
    ] + select({
        "//tensorflow/lite:tflite_use_simple_memory_planner": [
            "//tensorflow/lite:simple_planner",
        ],
        "//conditions:default": [
            "//tensorflow/lite:arena_planner",
        ],
    }) + select({
        "//tensorflow/lite:tensorflow_profiler_config": [
            "//tensorflow/lite:tensorflow_profiler_logger_shim",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,  # TODO(b/161243354): eliminate this.
)

# Test subgraph.
cc_test(
    name = "subgraph_test",
    size = "small",
    srcs = [
        "subgraph_test.cc",
    ],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)
