licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cuda_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

tf_cuda_library(
    name = "capture_profile",
    srcs = [
        "capture_profile.cc",
    ],
    hdrs = [
        "capture_profile.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":dump_tpu_profile",
        "//tensorflow:grpc++",
        "//tensorflow/core:framework",
        "//tensorflow/core:grpc_services",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/profiler:protos_all_cc",
    ],
)

cc_library(
    name = "dump_tpu_profile",
    srcs = ["dump_tpu_profile.cc"],
    hdrs = ["dump_tpu_profile.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":trace_events_to_json",
        "//tensorflow/core:framework",
        "//tensorflow/core:grpc_services",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:protos_all_cc",
    ],
)

cc_library(
    name = "trace_events_to_json",
    srcs = ["trace_events_to_json.cc"],
    hdrs = ["trace_events_to_json.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:protos_all_cc",
        "@jsoncpp_git//:jsoncpp",
    ],
)

tf_cc_test(
    name = "trace_events_to_json_test",
    srcs = ["trace_events_to_json_test.cc"],
    deps = [
        ":trace_events_to_json",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler:protos_all_cc",
        "@jsoncpp_git//:jsoncpp",
    ],
)
