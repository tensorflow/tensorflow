# Description:
#   OpKernels for RNN ops.

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_gpu_library",
)
load("//tensorflow:tensorflow.default.bzl", "tf_kernel_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])

tf_gpu_library(
    name = "blas_gemm",
    srcs = [] + if_cuda_is_configured([
        "blas_gemm.cc",
    ]) + if_rocm_is_configured([
        "blas_gemm.cc",
    ]),
    hdrs = ["blas_gemm.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core/kernels:numeric_options_utils",
        "//tensorflow/core/platform:stream_executor",
        "@eigen_archive//:eigen3",
        "@local_xla//xla/tsl/framework/contraction:eigen_contraction_kernel",
    ],
)

tf_kernel_library(
    name = "lstm_ops",
    prefix = "lstm_ops",
    deps = [
        ":blas_gemm",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:eigen_helpers",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "gru_ops",
    prefix = "gru_ops",
    deps = [
        ":blas_gemm",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:eigen_helpers",
        "@eigen_archive//:eigen3",
    ],
)
