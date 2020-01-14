package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

alias(
    name = "device_tracer",
    actual = "//tensorflow/core:device_tracer",
)

load(
    "//tensorflow:tensorflow.bzl",
    "if_cuda_is_configured_compat",
    "tf_copts",
    "tf_cuda_library",
)

tf_cuda_library(
    name = "cupti_interface",
    hdrs = if_cuda_is_configured_compat(["cupti_interface.h"]),
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:platform_base",
        "//tensorflow/stream_executor/cuda:cupti_stub",
        "@com_google_absl//absl/base:core_headers",
    ],
)

tf_cuda_library(
    name = "cupti_wrapper",
    srcs = if_cuda_is_configured_compat(["cupti_wrapper.cc"]),
    hdrs = if_cuda_is_configured_compat(["cupti_wrapper.h"]),
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
        "//tensorflow/stream_executor/cuda:cupti_stub",
    ],
)

tf_cuda_library(
    name = "cupti_tracer",
    srcs = if_cuda_is_configured_compat(["cupti_tracer.cc"]),
    hdrs = if_cuda_is_configured_compat(["cupti_tracer.h"]),
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:annotation",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/types:optional",
    ],
)
