load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "ynnpack_delegate",
    srcs = [
        "copy.cc",
        "copy.h",
        "dot.cc",
        "dot.h",
        "elementwise.cc",
        "elementwise.h",
        "pooling.cc",
        "pooling.h",
        "reduction.cc",
        "reduction.h",
        "softmax.cc",
        "softmax.h",
        "utils.cc",
        "utils.h",
        "ynnpack_delegate.cc",
    ],
    hdrs = ["ynnpack_delegate.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/core:subgraph",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/types:half",
        "@XNNPACK//ynnpack",
        "@XNNPACK//ynnpack/composites",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
        "@slinky//slinky/base:thread_pool",
        "@slinky//slinky/base:thread_pool_impl",
    ],
)

cc_test(
    name = "sdpa_test",
    srcs = ["sdpa_test.cc"],
    copts = tflite_copts(),
    deps = [
        ":ynnpack_delegate",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "sdpa_bench",
    srcs = ["sdpa_bench.cc"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":ynnpack_delegate",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/profiling:profile_summarizer",
        "//tensorflow/lite/profiling:profiler",
        "@com_google_benchmark//:benchmark",
    ],
)
