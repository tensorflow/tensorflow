# Description:
#   Wrap NVIDIA (https://github.com/NVIDIA/nccl) NCCL with tensorflow ops.
#   APIs are meant to change over time.

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_copts")
load("//tensorflow:tensorflow.bzl", "if_nccl")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)

cc_library(
    name = "nccl_lib",
    srcs = if_nccl([
        "nccl_manager.cc",
        "nccl_rewrite.cc",
    ]),
    hdrs = if_nccl([
        "nccl_manager.h",
    ]),
    copts = tf_copts(),
    deps = if_nccl([
        "@local_config_nccl//:nccl",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor",
    ]),
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "nccl_manager_test",
    size = "medium",
    srcs = ["nccl_manager_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "no_cuda_on_cpu_tap",  # TODO(b/120284216): re-enable multi_gpu
    ],
    deps = [
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ] + if_nccl([
        ":nccl_lib",
        "@local_config_nccl//:nccl",
    ]) + if_cuda_is_configured([
        "//tensorflow/core:cuda",
    ]) + if_rocm_is_configured([
        "//tensorflow/core:rocm",
    ]),
)
