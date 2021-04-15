# Description:
#   Wrap NVIDIA (https://github.com/NVIDIA/nccl) NCCL with tensorflow ops.
#   APIs are meant to change over time.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow:tensorflow.bzl", "if_cuda_or_rocm", "tf_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "if_nccl")

package(
    default_visibility = ["//tensorflow:__subpackages__"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

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
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:annotated_traceme",
    ]),
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "nccl_manager_test",
    size = "medium",
    srcs = ["nccl_manager_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "noguitar",  # TODO(b/176867216): Flaky.
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

cc_library(
    name = "collective_communicator",
    srcs = ["collective_communicator.cc"],
    hdrs = ["collective_communicator.h"],
    copts = tf_copts() + if_nccl(["-DTENSORFLOW_USE_NCCL=1"]),
    visibility = [
        "//learning/brain/runtime:__subpackages__",
        "//tensorflow:__subpackages__",
    ],
    deps =
        ["//tensorflow/core:framework"] + if_nccl([
            ":nccl_lib",
            "@com_google_absl//absl/memory",
            "//tensorflow/core/profiler/lib:traceme",
        ]),
)

filegroup(
    name = "mobile_srcs",
    srcs = [
        "collective_communicator.cc",
        "collective_communicator.h",
    ],
)
