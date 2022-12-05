# Description:
#   ROCm-platform specific StreamExecutor support code.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/compiler/xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "rocm_diagnostics",
    hdrs = if_rocm_is_configured(["rocm_diagnostics.h"]),
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:rocm_diagnostics"],
)

cc_library(
    name = "rocm_driver",
    hdrs = if_rocm_is_configured(["rocm_driver_wrapper.h"]),
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:rocm_driver"],
)

cc_library(
    name = "rocm_activation",
    hdrs = if_rocm_is_configured(["rocm_activation.h"]),
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:rocm_activation"],
)

cc_library(
    name = "rocm_event",
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:rocm_event"],
)

cc_library(
    name = "rocm_gpu_executor",
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:rocm_gpu_executor"],
)

cc_library(
    name = "rocm_kernel",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:rocm_kernel"],
)

cc_library(
    name = "rocm_platform",
    hdrs = if_rocm_is_configured(["rocm_platform.h"]),
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:rocm_platform"],
)

cc_library(
    name = "rocm_platform_id",
    hdrs = ["rocm_platform_id.h"],
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:rocm_platform_id"],
)

cc_library(
    name = "rocblas_if_static",
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:rocblas_if_static"],
)

cc_library(
    name = "rocblas_wrapper",
    hdrs = if_rocm_is_configured(["rocblas_wrapper.h"]),
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:rocblas_wrapper"],
)

cc_library(
    name = "rocblas_plugin",
    hdrs = if_rocm_is_configured(["rocm_blas.h"]),
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:rocblas_plugin"],
)

cc_library(
    name = "hipfft_if_static",
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:hipfft_if_static"],
)

cc_library(
    name = "hipfft_plugin",
    hdrs = if_rocm_is_configured(["rocm_fft.h"]),
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:hipfft_plugin"],
)

cc_library(
    name = "miopen_if_static",
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:miopen_if_static"],
)

cc_library(
    name = "miopen_plugin",
    hdrs = if_rocm_is_configured(["rocm_dnn.h"]),
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:miopen_plugin"],
)

cc_library(
    name = "hiprand_if_static",
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:hiprand_if_static"],
)

cc_library(
    name = "rocrand_plugin",
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:rocrand_plugin"],
)

cc_library(
    name = "hipsparse_if_static",
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:hipsparse_if_static"],
)

cc_library(
    name = "hipsparse_wrapper",
    hdrs = if_rocm_is_configured(["hipsparse_wrapper.h"]),
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:hipsparse_wrapper"],
)

cc_library(
    name = "rocsolver_if_static",
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:rocsolver_if_static"],
)

cc_library(
    name = "rocsolver_wrapper",
    hdrs = if_rocm_is_configured(["rocsolver_wrapper.h"]),
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:rocsolver_wrapper"],
)

cc_library(
    name = "hipsolver_if_static",
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:hipsolver_if_static"],
)

cc_library(
    name = "hipsolver_wrapper",
    hdrs = if_rocm_is_configured(["hipsolver_wrapper.h"]),
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:hipsolver_wrapper"],
)

cc_library(
    name = "roctracer_if_static",
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:roctracer_if_static"],
)

cc_library(
    name = "roctracer_wrapper",
    hdrs = if_rocm_is_configured(["roctracer_wrapper.h"]),
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:roctracer_wrapper"],
)

cc_library(
    name = "rocm_helpers",
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:rocm_helpers"],
)

cc_library(
    name = "all_runtime",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/compiler/xla/stream_executor/rocm:all_runtime"],
)
