# Description:
#   CUDA-platform specific StreamExecutor support code.

load("//tensorflow:tensorflow.bzl", "check_deps", "tf_copts")
load(
    "//tensorflow/compiler/xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
    "tf_additional_cudnn_plugin_copts",
)
load(
    "//tensorflow/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "cuda_platform_id",
    hdrs = ["cuda_platform_id.h"],
    deps = ["//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform_id"],
)

cc_library(
    name = "cuda_platform",
    hdrs = if_cuda_is_configured(["cuda_platform.h"]),
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_activation",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform",
    ],
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "cuda_diagnostics",
    hdrs = if_cuda_is_configured(["cuda_diagnostics.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_diagnostics",
    ],
)

# Buildozer can not remove dependencies inside select guards, so we have to use
# an intermediate target.
cc_library(name = "ptxas_wrapper")

# Buildozer can not remove dependencies inside select guards, so we have to use
# an intermediate target.
cc_library(name = "fatbinary_wrapper")

cc_library(
    name = "cuda_driver",
    hdrs = if_cuda_is_configured(["cuda_driver.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_driver",
    ],
)

# The activation library is tightly coupled to the executor library.
# TODO(leary) split up cuda_gpu_executor.cc so that this can stand alone.
cc_library(
    name = "cuda_activation_header",
    hdrs = ["cuda_activation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_activation_header",
    ],
)

cc_library(
    name = "cuda_activation",
    hdrs = if_cuda_is_configured(["cuda_activation.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_activation",
    ],
)

cc_library(
    name = "cuda_gpu_executor_header",
    textual_hdrs = if_cuda_is_configured(["cuda_gpu_executor.h"]),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_gpu_executor_header",
    ],
)

cc_library(
    name = "cublas_stub",
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cublas_stub",
    ],
)

alias(
    name = "cublas_lib",
    actual = select({
        "//tensorflow:oss": ":cublas_stub",
        "//conditions:default": "//third_party/gpus/cuda:cublas_static",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cublas_lt_header",
    hdrs = if_cuda_is_configured([
        "cuda_blas_lt.h",
        "cuda_blas_utils.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cublas_lt_header",
    ],
)

cc_library(
    name = "cublas_lt_stub",
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cublas_lt_stub",
    ],
)

alias(
    name = "cublas_lt_lib",
    actual = select({
        "//tensorflow:oss": ":cublas_lt_stub",
        "//conditions:default": "//third_party/gpus/cuda:cublas_lt_static",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cublas_plugin",
    hdrs = if_cuda_is_configured([
        "cuda_blas.h",
        "cuda_blas_lt.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cublas_plugin",
    ],
    alwayslink = True,
)

cc_library(
    name = "cuda_blas_utils",
    hdrs = if_cuda_is_configured(["cuda_blas_utils.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_blas_utils",
    ],
)

cc_library(
    name = "cufft_stub",
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cufft_stub",
    ],
)

alias(
    name = "cufft_lib",
    actual = select({
        "//tensorflow:oss": ":cufft_stub",
        "//conditions:default": "//third_party/gpus/cuda:cufft_static",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cufft_plugin",
    hdrs = if_cuda_is_configured(["cuda_fft.h"]),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cufft_plugin",
    ],
    alwayslink = True,
)

cc_library(
    name = "cudnn_stub",
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cudnn_stub",
    ],
)

alias(
    name = "cudnn_lib",
    actual = select({
        "//tensorflow:oss": ":cudnn_stub",
        "//conditions:default": "//third_party/gpus/cudnn:cudnn",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_dnn_headers",
    textual_hdrs = ["cuda_dnn.h"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_dnn_headers",
    ],
)

cc_library(
    name = "cudnn_plugin",
    hdrs = if_cuda_is_configured(["cuda_dnn.h"]),
    copts = tf_additional_cudnn_plugin_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cudnn_plugin",
    ],
    alwayslink = True,
)

cc_library(
    name = "curand_stub",
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:curand_stub",
    ],
)

alias(
    name = "curand_lib",
    actual = select({
        "//tensorflow:oss": ":curand_stub",
        "//conditions:default": "//third_party/gpus/cuda:curand_static",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "curand_plugin",
    hdrs = if_cuda_is_configured(["cuda_rng.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:curand_plugin",
    ],
    alwayslink = True,
)

cc_library(
    name = "cupti_stub",
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cupti_stub",
    ],
)

cc_library(
    name = "cusolver_stub",
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cusolver_stub",
    ],
)

alias(
    name = "cusolver_lib",
    actual = select({
        "//tensorflow:oss": ":cusolver_stub",
        "//conditions:default": "//third_party/gpus/cuda:cusolver_static",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cusparse_stub",
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cusparse_stub",
    ],
)

alias(
    name = "cusparse_lib",
    actual = select({
        "//tensorflow:oss": ":cusparse_stub",
        "//conditions:default": "//third_party/gpus/cuda:cusparse_static",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_kernel",
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_kernel",
    ],
)

# TODO(leary) we likely need to canonicalize/eliminate this.
cc_library(
    name = "cuda_helpers",
    textual_hdrs = if_cuda_is_configured(["cuda_helpers.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_helpers",
    ],
)

cc_library(
    name = "cuda_event",
    hdrs = if_cuda_is_configured(["cuda_event.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_event",
    ],
)

cc_library(
    name = "cuda_stream",
    hdrs = if_cuda_is_configured(["cuda_stream.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_stream",
    ],
)

cc_library(
    name = "cuda_timer",
    hdrs = if_cuda_is_configured(["cuda_timer.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_timer",
    ],
)

cc_library(
    name = "cuda_asm_compiler",
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_asm_compiler",
    ],
)

cc_library(
    name = "cuda_gpu_executor",
    hdrs = if_cuda_is_configured(["cuda_gpu_executor.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_gpu_executor",
    ],
    alwayslink = True,
)

cc_library(
    name = "all_runtime",
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:all_runtime",
    ],
    alwayslink = 1,
)

# To avoid duplication, check that the C++ or python library does not depend on
# the stream executor cuda plugins. Targets that want to use cuda APIs should
# instead depend on the dummy plugins in //tensorflow/tsl/platform/default/build_config
# and use header only targets.
check_deps(
    name = "cuda_plugins_check_deps",
    disallowed_deps = if_static(
        [],
        otherwise = [
            ":all_runtime",
            ":cuda_driver",
            ":cuda_platform",
            ":cudnn_plugin",
            ":cufft_plugin",
            ":curand_plugin",
            "//tensorflow/stream_executor:cuda_platform",
        ],
    ),
    deps = [
        "//tensorflow:tensorflow_cc",
        "//tensorflow/python:pywrap_tensorflow_internal",
    ],
)
