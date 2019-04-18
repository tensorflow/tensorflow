package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/lite:build_def.bzl", "tflite_copts")

common_copts = [
    "-Wall",
] + tflite_copts()

cc_library(
    name = "profiler",
    hdrs = ["profiler.h"],
    copts = common_copts,
    deps = [":profile_buffer"],
)

cc_test(
    name = "profiler_test",
    srcs = ["profiler_test.cc"],
    copts = ["-DTFLITE_PROFILING_ENABLED"],
    defines = ["TFLITE_PROFILING_ENABLED"],
    deps = [
        ":profiler",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "profile_buffer",
    hdrs = ["profile_buffer.h"],
    copts = common_copts,
    deps = [":time"],
)

cc_library(
    name = "time",
    srcs = ["time.cc"],
    hdrs = ["time.h"],
    copts = common_copts,
)

cc_test(
    name = "time_test",
    srcs = ["time_test.cc"],
    copts = common_copts,
    deps = [
        ":time",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "profile_summarizer",
    srcs = ["profile_summarizer.cc"],
    hdrs = ["profile_summarizer.h"],
    copts = common_copts,
    deps = [
        ":profiler",
        "//tensorflow/core:stats_calculator_portable",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "profile_summarizer_test",
    srcs = ["profile_summarizer_test.cc"],
    copts = common_copts,
    deps = [
        ":profile_summarizer",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "profile_buffer_test",
    srcs = ["profile_buffer_test.cc"],
    copts = ["-DTFLITE_PROFILING_ENABLED"],
    defines = ["TFLITE_PROFILING_ENABLED"],
    deps = [
        ":profile_buffer",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)
