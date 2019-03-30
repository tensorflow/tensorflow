package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:build_def.bzl", "tflite_linkopts")

common_copts = ["-Wall"] + tflite_copts()

cc_library(
    name = "logging",
    hdrs = ["logging.h"],
    copts = common_copts,
)

cc_binary(
    name = "benchmark_model",
    srcs = [
        "benchmark_main.cc",
    ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":benchmark_tflite_model_lib",
        ":logging",
    ],
)

tf_cc_binary(
    name = "benchmark_model_plus_flex",
    srcs = [
        "benchmark_plus_flex_main.cc",
    ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":benchmark_tflite_model_lib",
        ":logging",
        "//tensorflow/lite/delegates/flex:delegate",
        "//tensorflow/lite/testing:init_tensorflow",
    ],
)

cc_test(
    name = "benchmark_test",
    srcs = ["benchmark_test.cc"],
    args = [
        "--graph=$(location //tensorflow/lite:testdata/multi_add.bin)",
    ],
    data = ["//tensorflow/lite:testdata/multi_add.bin"],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":benchmark_tflite_model_lib",
        ":command_line_flags",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "command_line_flags",
    srcs = ["command_line_flags.cc"],
    hdrs = ["command_line_flags.h"],
    copts = common_copts,
)

cc_test(
    name = "command_line_flags_test",
    srcs = ["command_line_flags_test.cc"],
    copts = common_copts,
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":command_line_flags",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "benchmark_tflite_model_lib",
    srcs = [
        "benchmark_tflite_model.cc",
        "logging.h",
    ],
    hdrs = ["benchmark_tflite_model.h"],
    copts = common_copts,
    deps = [
        ":benchmark_model_lib",
        ":logging",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/profiling:profile_summarizer",
        "@gemmlowp",
    ],
)

cc_library(
    name = "benchmark_params",
    srcs = [
        "benchmark_params.cc",
    ],
    hdrs = ["benchmark_params.h"],
    copts = common_copts,
    deps = [":logging"],
)

cc_library(
    name = "benchmark_model_lib",
    srcs = [
        "benchmark_model.cc",
    ],
    hdrs = ["benchmark_model.h"],
    copts = common_copts,
    deps = [
        ":benchmark_params",
        ":command_line_flags",
        ":logging",
        "//tensorflow/core:stats_calculator_portable",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/profiling:time",
    ],
)

tflite_portable_test_suite()
