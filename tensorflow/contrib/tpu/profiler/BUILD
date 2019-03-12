licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "version",
    hdrs = ["version.h"],
    visibility = ["//visibility:public"],
)

tf_cc_binary(
    name = "capture_tpu_profile",
    srcs = [
        "capture_tpu_profile.cc",
    ],
    tags = ["no_windows"],
    visibility = ["//visibility:public"],
    deps = [
        ":version",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform/cloud:gcs_file_system",
        "//tensorflow/core/profiler/rpc/client:capture_profile",
    ],
)
