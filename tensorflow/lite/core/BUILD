load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    licenses = ["notice"],
)

exports_files([
    "macros.h",
    "subgraph.h",
])

# WARNING: Deps of :headers and :private_headers must be the same.
# If you update deps of :headers, then make the same update to :private_headers.
cc_library(
    name = "headers",
    hdrs = [
        "interpreter.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":subgraph",
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:external_cpu_backend_context",
        "//tensorflow/lite:interpreter_options_header",
        "//tensorflow/lite:signature_runner_header",
        "//tensorflow/lite:stderr_reporter",
        "//tensorflow/lite:string",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/experimental/remat:metadata_util",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/internal:signature_def",
        "//tensorflow/lite/profiling:root_profiler",
    ],
)

# This target is only for use by the 'tflite_custom_c_library' build macro
# and should not be used anywhere other than in the implementation of that build macro.
# 'tflite_custom_c_library' requires target to be public, that's why we duplicated :headers target
# to be used only by 'tflite_custom_c_library'.
# Making :headers public could cause some problems because it is widely used inside the TF Lite
# code base, that might lead others outside the TF Lite code base to copy that dependency and use
# it and subsequently depend on it, which would be bad. Using a separate :private_headers target
# ensures that the only use of the unwantedly-"public" target is inside the
# 'tflite_custom_c_library' itself, where it is less likely to get copied into third party code.
# WARNING: Deps of :private_headers and :headers must be the same.
# If you update deps of :private_headers, then make the same update to :headers.
cc_library(
    name = "private_headers",
    hdrs = [
        "interpreter.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":subgraph",
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:external_cpu_backend_context",
        "//tensorflow/lite:interpreter_options_header",
        "//tensorflow/lite:signature_runner_header",
        "//tensorflow/lite:stderr_reporter",
        "//tensorflow/lite:string",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/experimental/remat:metadata_util",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/internal:signature_def",
        "//tensorflow/lite/profiling:root_profiler",
    ],
)

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
        "//tensorflow/lite/experimental/remat:metadata_util",
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
        ":headers",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)
