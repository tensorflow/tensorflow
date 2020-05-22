# Description:
#   Wrap NVIDIA (https://github.com/NVIDIA/nccl) NCCL with tensorflow ops.
#   APIs are meant to change over time.

load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("//tensorflow:tensorflow.bzl", "if_cuda_or_rocm")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

cc_library(
    name = "nccl_lib",
    srcs = if_cuda_or_rocm([
        "nccl_manager.cc",
        "nccl_rewrite.cc",
    ]),
    hdrs = if_cuda_or_rocm([
        "nccl_manager.h",
    ]),
    copts = tf_copts(),
    deps = if_cuda([
        "@local_config_nccl//:nccl",
    ]) + if_rocm([
        "@local_config_rocm//rocm:rccl",
        "//tensorflow/core:gpu_runtime",
    ]) + if_cuda_or_rocm([
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor",
        "//tensorflow/core/profiler/lib:traceme",
    ]),
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "nccl_manager_test",
    size = "medium",
    srcs = ["nccl_manager_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "guitar",
        "manual",
        "multi_gpu",
        "no_oss",
        # TODO(b/147451637): Replace 'no_rocm' with 'rocm_multi_gpu'.
        "no_rocm",
        "notap",
    ],
    deps = [
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ] + if_cuda_or_rocm([
        ":nccl_lib",
    ]) + if_cuda([
        "@local_config_nccl//:nccl",
        "//tensorflow/core:cuda",
    ]) + if_rocm([
        "@local_config_rocm//rocm:rccl",
        "//tensorflow/core/common_runtime/gpu:rocm",
    ]),
)
