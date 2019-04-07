# Description:
#   GPU-platform specific StreamExecutor support code.

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")

package(
    default_visibility = ["//tensorflow/stream_executor:__subpackages__"],
)

# Filegroup used to collect source files for the dependency check.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "gpu_activation_header",
    hdrs = ["gpu_activation.h"],
    deps = ["//tensorflow/stream_executor/platform"],
)

cc_library(
    name = "gpu_activation",
    srcs = if_gpu_is_configured(["gpu_activation.cc"]),
    hdrs = if_gpu_is_configured(["gpu_activation.h"]),
    deps = if_gpu_is_configured([
        ":gpu_activation_header",
        ":gpu_driver_header",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor:stream_executor_internal",
        "//tensorflow/stream_executor/platform",
    ]),
)

cc_library(
    name = "gpu_diagnostics_header",
    hdrs = if_gpu_is_configured(["gpu_diagnostics.h"]),
    deps = [
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "gpu_driver_header",
    hdrs = if_gpu_is_configured(["gpu_driver.h"]),
    deps = [
        ":gpu_types_header",
        "//tensorflow/stream_executor:device_options",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "gpu_event_header",
    hdrs = if_gpu_is_configured(["gpu_event.h"]),
    deps = [
        ":gpu_driver_header",
        ":gpu_stream_header",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor/lib",
    ],
)

cc_library(
    name = "gpu_event",
    srcs = if_gpu_is_configured(["gpu_event.cc"]),
    hdrs = if_gpu_is_configured(["gpu_event.h"]),
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        ":gpu_stream",
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor/lib",
    ],
)

cc_library(
    name = "gpu_executor_header",
    hdrs = if_gpu_is_configured(["gpu_executor.h"]),
    deps = [
        ":gpu_kernel_header",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:platform",
        "//tensorflow/stream_executor:stream_executor_internal",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_helpers_header",
    hdrs = if_gpu_is_configured(["gpu_helpers.h"]),
    deps = [":gpu_types_header"],
)

cc_library(
    name = "gpu_kernel_header",
    hdrs = if_gpu_is_configured(["gpu_kernel.h"]),
    deps = [
        ":gpu_driver_header",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "gpu_rng_header",
    hdrs = if_gpu_is_configured(["gpu_rng.h"]),
    deps = [
        ":gpu_types_header",
        "//tensorflow/stream_executor:plugin_registry",
        "//tensorflow/stream_executor:rng",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "gpu_stream_header",
    hdrs = if_gpu_is_configured(["gpu_stream.h"]),
    deps = [
        ":gpu_driver_header",
        "//tensorflow/stream_executor:stream_executor_internal",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "gpu_stream",
    srcs = if_gpu_is_configured(["gpu_stream.cc"]),
    hdrs = if_gpu_is_configured(["gpu_stream.h"]),
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor:stream_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "gpu_timer_header",
    hdrs = if_gpu_is_configured(["gpu_timer.h"]),
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        "//tensorflow/stream_executor:stream_executor_internal",
    ],
)

cc_library(
    name = "gpu_timer",
    srcs = if_gpu_is_configured(["gpu_timer.cc"]),
    hdrs = if_gpu_is_configured(["gpu_timer.h"]),
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        ":gpu_stream",
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor/lib",
    ],
)

cc_library(
    name = "gpu_types_header",
    hdrs = if_gpu_is_configured(["gpu_types.h"]),
    deps = [
        "//tensorflow/stream_executor/platform",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)
