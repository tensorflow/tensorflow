# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Description:
#   OpKernels for RNN ops.

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
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
        "@xla//xla/tsl/framework/contraction:eigen_contraction_kernel",
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
)
