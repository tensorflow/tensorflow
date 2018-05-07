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
