load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

common_copts = [
    "-Wall",
] + tflite_copts()

cc_library(
    name = "profiler",
    hdrs = [
        "buffered_profiler.h",
        "noop_profiler.h",
        "profiler.h",
    ],
    copts = common_copts,
    deps = [
        ":profile_buffer",
        "//tensorflow/lite/core/api",
    ],
)

cc_test(
    name = "profiler_test",
    srcs = ["profiler_test.cc"],
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
    deps = [
        ":memory_info",
        ":time",
        "//tensorflow/lite/core/api",
    ],
)

cc_library(
    name = "time",
    srcs = ["time.cc"],
    hdrs = ["time.h"],
    copts = common_copts,
)

cc_library(
    name = "memory_info",
    srcs = ["memory_info.cc"],
    hdrs = ["memory_info.h"],
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

cc_test(
    name = "memory_info_test",
    srcs = ["memory_info_test.cc"],
    copts = common_copts,
    tags = [
        # Some low-level checks, like heap size check, may break in asan, msan
        # and tsan. So, disable such tests.
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        ":memory_info",
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
        ":memory_info",
        ":profile_buffer",
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
        ":profiler",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:subgraph_test_util",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "profile_buffer_test",
    srcs = ["profile_buffer_test.cc"],
    deps = [
        ":profile_buffer",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)
