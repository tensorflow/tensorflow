package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/contrib/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow/contrib/lite:build_def.bzl", "tflite_linkopts")
load("//tensorflow/contrib/lite:build_def.bzl", "tflite_copts")

common_copts = ["-Wall"] + tflite_copts()

cc_binary(
    name = "benchmark_model",
    srcs = [
        "benchmark_main.cc",
        "logging.h",
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
    visibility = ["//visibility:private"],
    deps = [
        ":command_line_flags",
        "//tensorflow/contrib/lite/testing:util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:string_util",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
        "//tensorflow/contrib/lite/profiling:profile_summarizer",
        "//tensorflow/contrib/lite/profiling:profiler",
    ],
)

cc_library(
    name = "benchmark_params",
    srcs = [
        "benchmark_params.cc",
        "logging.h",
    ],
    hdrs = ["benchmark_params.h"],
    copts = common_copts,
)

cc_library(
    name = "benchmark_model_lib",
    srcs = [
        "benchmark_model.cc",
        "logging.h",
    ],
    hdrs = ["benchmark_model.h"],
    copts = common_copts,
    deps = [
        ":benchmark_params",
        ":command_line_flags",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:string_util",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
        "//tensorflow/contrib/lite/profiling:profile_summarizer",
        "//tensorflow/contrib/lite/profiling:profiler",
        "//tensorflow/contrib/lite/profiling:time",
        "//tensorflow/core:stats_calculator_portable",
    ],
)

tflite_portable_test_suite()
