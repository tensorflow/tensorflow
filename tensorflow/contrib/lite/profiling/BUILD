package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

common_copts = [
    "-Wall",
]

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
        "//tensorflow/contrib/lite/testing:util",
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
)

cc_library(
    name = "profile_summarizer",
    srcs = ["profile_summarizer.cc"],
    hdrs = ["profile_summarizer.h"],
    deps = [
        ":profiler",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/schema:schema_fbs",
        "//tensorflow/core:stats_calculator_portable",
    ],
)

cc_test(
    name = "profile_summarizer_test",
    srcs = ["profile_summarizer_test.cc"],
    deps = [
        ":profile_summarizer",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:schema_fbs_version",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
        "//tensorflow/contrib/lite/kernels:kernel_util",
        "//tensorflow/contrib/lite/kernels:test_util",
        "//tensorflow/contrib/lite/testing:util",
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
        "//tensorflow/contrib/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)
