package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "if_not_windows", "tf_cc_test")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

exports_files(glob([
    "testdata/*.bin",
    "testdata/*.pb",
    "models/testdata/*",
]))

config_setting(
    name = "mips",
    values = {
        "cpu": "mips",
    },
)

config_setting(
    name = "mips64",
    values = {
        "cpu": "mips64",
    },
)

# Enables inclusion of select TensorFlow kernels via the TF Lite Flex delegate.
# WARNING: This build flag is experimental and subject to change.
config_setting(
    name = "with_select_tf_ops",
    define_values = {"with_select_tf_ops": "true"},
    visibility = ["//visibility:public"],
)

TFLITE_DEFAULT_COPTS = if_not_windows([
    "-Wall",
    "-Wno-comment",
])

cc_library(
    name = "schema_fbs_version",
    hdrs = ["version.h"],
    copts = TFLITE_DEFAULT_COPTS,
)

cc_library(
    name = "arena_planner",
    srcs = ["arena_planner.cc"],
    hdrs = ["arena_planner.h"],
    copts = TFLITE_DEFAULT_COPTS,
    deps = [
        ":graph_info",
        ":memory_planner",
        ":simple_memory_arena",
        "//tensorflow/lite/c:c_api_internal",
    ],
)

tf_cc_test(
    name = "arena_planner_test",
    size = "small",
    srcs = ["arena_planner_test.cc"],
    deps = [
        ":arena_planner",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Main library. No ops are included here.
# TODO(aselle): Resolve problems preventing C99 usage.
cc_library(
    name = "context",
    hdrs = ["context.h"],
    copts = TFLITE_DEFAULT_COPTS,
    deps = ["//tensorflow/lite/c:c_api_internal"],
)

cc_library(
    name = "graph_info",
    hdrs = ["graph_info.h"],
    copts = TFLITE_DEFAULT_COPTS,
    deps = ["//tensorflow/lite/c:c_api_internal"],
)

cc_library(
    name = "memory_planner",
    hdrs = ["memory_planner.h"],
    copts = TFLITE_DEFAULT_COPTS,
    deps = ["//tensorflow/lite/c:c_api_internal"],
)

cc_library(
    name = "simple_memory_arena",
    srcs = ["simple_memory_arena.cc"],
    hdrs = ["simple_memory_arena.h"],
    copts = TFLITE_DEFAULT_COPTS,
    deps = ["//tensorflow/lite/c:c_api_internal"],
)

cc_library(
    name = "builtin_op_data",
    hdrs = [
        "builtin_op_data.h",
    ],
    deps = ["//tensorflow/lite/c:c_api_internal"],
)

cc_library(
    name = "kernel_api",
    hdrs = [
        "builtin_op_data.h",
        "builtin_ops.h",
        "context_util.h",
    ],
    deps = ["//tensorflow/lite/c:c_api_internal"],
)

exports_files(["builtin_ops.h"])

cc_library(
    name = "string",
    hdrs = [
        "string.h",
    ],
    copts = TFLITE_DEFAULT_COPTS,
)

# TODO(ahentz): investigate dependency on gemm_support requiring usage of tf_copts.
cc_library(
    name = "framework",
    srcs = [
        "allocation.cc",
        "core/subgraph.cc",
        "graph_info.cc",
        "interpreter.cc",
        "model.cc",
        "mutable_op_resolver.cc",
        "optional_debug_tools.cc",
        "stderr_reporter.cc",
    ] + select({
        "//tensorflow:android": [
            "nnapi_delegate.cc",
            "mmap_allocation.cc",
        ],
        "//tensorflow:windows": [
            "nnapi_delegate_disabled.cc",
            "mmap_allocation_disabled.cc",
        ],
        "//conditions:default": [
            "nnapi_delegate_disabled.cc",
            "mmap_allocation.cc",
        ],
    }),
    hdrs = [
        "allocation.h",
        "context.h",
        "context_util.h",
        "core/subgraph.h",
        "error_reporter.h",
        "graph_info.h",
        "interpreter.h",
        "model.h",
        "mutable_op_resolver.h",
        "nnapi_delegate.h",
        "op_resolver.h",
        "optional_debug_tools.h",
        "stderr_reporter.h",
    ],
    copts = tflite_copts() + TFLITE_DEFAULT_COPTS,
    linkopts = [
    ] + select({
        "//tensorflow:android": [
            "-llog",
        ],
        "//conditions:default": [
        ],
    }),
    deps = [
        ":arena_planner",
        ":graph_info",
        ":memory_planner",
        ":schema_fbs_version",
        ":simple_memory_arena",
        ":string",
        ":util",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "//tensorflow/lite/profiling:profiler",
        "//tensorflow/lite/schema:schema_fbs",
    ] + select({
        ":with_select_tf_ops": [
            "//tensorflow/lite/delegates/flex:delegate",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "string_util",
    srcs = ["string_util.cc"],
    hdrs = ["string_util.h"],
    copts = TFLITE_DEFAULT_COPTS,
    deps = [
        ":string",
        "//tensorflow/lite/c:c_api_internal",
    ],
)

cc_test(
    name = "string_util_test",
    size = "small",
    srcs = ["string_util_test.cc"],
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":framework",
        ":string_util",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Test main interpreter
cc_test(
    name = "interpreter_test",
    size = "small",
    srcs = ["interpreter_test.cc"],
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":framework",
        ":string_util",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Test graph utils
cc_test(
    name = "graph_info_test",
    size = "small",
    srcs = ["graph_info_test.cc"],
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":framework",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Test arena allocator
cc_test(
    name = "simple_memory_arena_test",
    size = "small",
    srcs = ["simple_memory_arena_test.cc"],
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":simple_memory_arena",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Test model framework.
cc_test(
    name = "model_test",
    size = "small",
    srcs = ["model_test.cc"],
    data = [
        "testdata/0_subgraphs.bin",
        "testdata/2_subgraphs.bin",
        "testdata/empty_model.bin",
        "testdata/multi_add_flex.bin",
        "testdata/test_model.bin",
        "testdata/test_model_broken.bin",
    ],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Test model framework with the flex library linked into the target.
tf_cc_test(
    name = "model_flex_test",
    size = "small",
    srcs = ["model_flex_test.cc"],
    data = [
        "testdata/multi_add_flex.bin",
    ],
    tags = [
        "no_gpu",  # GPU + flex is not officially supported.
        "no_windows",  # TODO(b/116667551): No weak symbols with MSVC.
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/delegates/flex:delegate",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Test OpResolver.
cc_test(
    name = "mutable_op_resolver_test",
    size = "small",
    srcs = ["mutable_op_resolver_test.cc"],
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":framework",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    copts = TFLITE_DEFAULT_COPTS + tflite_copts(),
    deps = [
        "//tensorflow/lite/c:c_api_internal",
    ],
)

cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":util",
        "//tensorflow/lite/c:c_api_internal",
        "@com_google_googletest//:gtest",
    ],
)

tflite_portable_test_suite()
