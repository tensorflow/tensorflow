package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/contrib/lite:build_def.bzl", "tflite_copts", "gen_selected_ops")

exports_files(glob([
    "testdata/*.bin",
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

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

cc_library(
    name = "schema_fbs_version",
    hdrs = ["version.h"],
)

cc_library(
    name = "arena_planner",
    srcs = ["arena_planner.cc"],
    hdrs = ["arena_planner.h"],
    deps = [
        ":context",
        ":graph_info",
        ":memory_planner",
        ":simple_memory_arena",
    ],
)

cc_test(
    name = "arena_planner_test",
    size = "small",
    srcs = ["arena_planner_test.cc"],
    deps = [
        ":arena_planner",
        "//tensorflow/contrib/lite/testing:util",
        "//tensorflow/core:lib",
        "@com_google_googletest//:gtest",
    ],
)

# Main library. No ops are included here.
# TODO(aselle): Resolve problems preventing C99 usage.
cc_library(
    name = "context",
    srcs = ["context.c"],
    hdrs = ["context.h"],
)

cc_library(
    name = "graph_info",
    hdrs = ["graph_info.h"],
    deps = [":context"],
)

cc_library(
    name = "memory_planner",
    hdrs = ["memory_planner.h"],
    deps = [":context"],
)

cc_library(
    name = "simple_memory_arena",
    srcs = ["simple_memory_arena.cc"],
    hdrs = ["simple_memory_arena.h"],
    deps = [":context"],
)

cc_library(
    name = "builtin_op_data",
    hdrs = [
        "builtin_op_data.h",
    ],
)

cc_library(
    name = "string",
    hdrs = [
        "string.h",
    ],
    deps = [
        "//tensorflow/core:lib_platform",
    ],
)

# TODO(ahentz): investigate dependency on gemm_support requiring usage of tf_copts.
cc_library(
    name = "framework",
    srcs = [
        "allocation.cc",
        "error_reporter.cc",
        "interpreter.cc",
        "model.cc",
        "nnapi_delegate.cc",
        "optional_debug_tools.cc",
    ],
    hdrs = [
        "allocation.h",
        "context.h",
        "error_reporter.h",
        "interpreter.h",
        "model.h",
        "nnapi_delegate.h",
        "optional_debug_tools.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":arena_planner",
        ":builtin_op_data",
        ":context",
        ":graph_info",
        ":memory_planner",
        ":schema_fbs_version",
        ":simple_memory_arena",
        "//tensorflow/contrib/lite/kernels:gemm_support",
        "//tensorflow/contrib/lite/nnapi:nnapi_lib",
        "//tensorflow/contrib/lite/schema:schema_fbs",
        "//tensorflow/core:lib_platform",
    ],
)

cc_library(
    name = "string_util",
    srcs = ["string_util.cc"],
    hdrs = ["string_util.h"],
    deps = [
        ":framework",
        ":string",
    ],
)

cc_test(
    name = "string_util_test",
    size = "small",
    srcs = ["string_util_test.cc"],
    deps = [
        ":framework",
        ":string_util",
        "//tensorflow/contrib/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Test main interpreter
cc_test(
    name = "interpreter_test",
    size = "small",
    srcs = ["interpreter_test.cc"],
    deps = [
        ":framework",
        ":string_util",
        "//tensorflow/contrib/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Test arena allocator
cc_test(
    name = "simple_memory_arena_test",
    size = "small",
    srcs = ["simple_memory_arena_test.cc"],
    deps = [
        ":simple_memory_arena",
        "//tensorflow/contrib/lite/testing:util",
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
        "testdata/test_model.bin",
        "testdata/test_model_broken.bin",
    ],
    deps = [
        ":framework",
        "//tensorflow/contrib/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Test the C extension API code.
cc_test(
    name = "context_test",
    size = "small",
    srcs = ["context_test.cc"],
    deps = [
        ":framework",
        "//tensorflow/contrib/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Test the serialization of a model with optional tensors.

# Model tests

cc_library(
    name = "models_test_utils",
    testonly = 1,
    hdrs = ["models/test_utils.h"],
    deps = select({
        "//tensorflow:android": [],
        "//conditions:default": [
            "@com_google_absl//absl/strings",
            "//tensorflow/core:test",
        ],
    }),
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "downloads",
            "examples",
            "gen",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
