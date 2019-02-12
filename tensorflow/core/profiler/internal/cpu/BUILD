package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cuda_library",
)
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")

tf_cuda_library(
    name = "host_tracer",
    srcs = [
        "host_tracer.cc",
    ],
    hdrs = [
        "host_tracer.h",
    ],
    deps = [
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler/internal:traceme_recorder",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "host_tracer_test",
    srcs = ["host_tracer_test.cc"],
    deps = [
        ":host_tracer",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
    ],
)
