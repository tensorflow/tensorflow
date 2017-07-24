licenses(["notice"])  # Apache 2.0

load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library_cc")

tf_proto_library_cc(
    name = "tpu_profiler_proto",
    srcs = ["tpu_profiler.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    protodeps = [
        "//tensorflow/core:protos_all",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "capture_tpu_profile",
    srcs = ["capture_tpu_profile.cc"],
    visibility = ["//tensorflow/contrib/tpu/profiler:__subpackages__"],
    deps = [
        ":tpu_profiler_proto_cc",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "@grpc//:grpc++_unsecure",
    ],
)

tf_proto_library(
    name = "trace_events_proto",
    srcs = ["trace_events.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)
