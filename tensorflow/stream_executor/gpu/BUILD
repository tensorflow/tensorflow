# Description:
#   GPU-platform specific StreamExecutor support code.

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

package(
    default_visibility = [":friends"],
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
    visibility = ["//tensorflow/stream_executor:__subpackages__"],
    deps = ["//tensorflow/stream_executor/platform"],
)

cc_library(
    name = "gpu_activation",
    srcs = ["gpu_activation.cc"],
    hdrs = ["gpu_activation.h"],
    visibility = ["//tensorflow/stream_executor:__subpackages__"],
    deps = [
        ":gpu_activation_header",
        ":gpu_driver_header",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor:stream_executor_internal",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "gpu_diagnostics_header",
    hdrs = ["gpu_diagnostics.h"],
    visibility = ["//tensorflow/stream_executor:__subpackages__"],
    deps = [
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "gpu_driver_header",
    hdrs = ["gpu_driver.h"],
    visibility = ["//tensorflow/stream_executor:__subpackages__"],
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
    hdrs = ["gpu_event.h"],
    visibility = ["//tensorflow/stream_executor:__subpackages__"],
    deps = [
        ":gpu_driver_header",
        ":gpu_stream_header",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor/lib",
    ],
)

cc_library(
    name = "gpu_event",
    srcs = ["gpu_event.cc"],
    hdrs = ["gpu_event.h"],
    visibility = ["//tensorflow/stream_executor:__subpackages__"],
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
    hdrs = ["gpu_executor.h"],
    visibility = ["//tensorflow/stream_executor:__subpackages__"],
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
    hdrs = ["gpu_helpers.h"],
    visibility = ["//tensorflow/stream_executor:__subpackages__"],
    deps = [":gpu_types_header"],
)

cc_library(
    name = "gpu_kernel_header",
    hdrs = ["gpu_kernel.h"],
    visibility = ["//tensorflow/stream_executor:__subpackages__"],
    deps = [
        ":gpu_driver_header",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "gpu_rng_header",
    hdrs = ["gpu_rng.h"],
    visibility = ["//tensorflow/stream_executor:__subpackages__"],
    deps = [
        ":gpu_types_header",
        "//tensorflow/stream_executor:plugin_registry",
        "//tensorflow/stream_executor:rng",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "gpu_stream_header",
    hdrs = ["gpu_stream.h"],
    visibility = ["//tensorflow/stream_executor:__subpackages__"],
    deps = [
        ":gpu_driver_header",
        "//tensorflow/stream_executor:stream_executor_internal",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "gpu_stream",
    srcs = ["gpu_stream.cc"],
    hdrs = ["gpu_stream.h"],
    visibility = ["//tensorflow/stream_executor:__subpackages__"],
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
    hdrs = ["gpu_timer.h"],
    visibility = ["//tensorflow/stream_executor:__subpackages__"],
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        "//tensorflow/stream_executor:stream_executor_internal",
    ],
)

cc_library(
    name = "gpu_timer",
    srcs = ["gpu_timer.cc"],
    hdrs = ["gpu_timer.h"],
    visibility = ["//tensorflow/stream_executor:__subpackages__"],
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
    hdrs = ["gpu_types.h"],
    visibility = ["//tensorflow/stream_executor:__subpackages__"],
    deps = [
        "//tensorflow/stream_executor/platform",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)
