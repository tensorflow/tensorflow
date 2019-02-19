licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cuda_library")

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
        "//tensorflow:grpc++",
        "//tensorflow/contrib/tpu/profiler:dump_tpu_profile",
        "//tensorflow/contrib/tpu/profiler:tpu_profiler_analysis_proto_cc",
        "//tensorflow/contrib/tpu/profiler:tpu_profiler_proto_cc",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/platform/cloud:gcs_file_system",
    ],
)
