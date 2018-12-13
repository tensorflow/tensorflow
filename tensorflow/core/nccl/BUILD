# Description:
#   Wrap NVIDIA (https://github.com/NVIDIA/nccl) NCCL with tensorflow ops.
#   APIs are meant to change over time.

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)

cc_library(
    name = "nccl_lib",
    srcs = if_cuda([
        "nccl_manager.cc",
        "nccl_manager.h",
        "nccl_rewrite.cc",
    ]),
    copts = tf_copts(),
    deps = if_cuda([
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
    ] + if_cuda([
        ":nccl_lib",
        "@local_config_nccl//:nccl",
        "//tensorflow/core:cuda",
    ]),
)
