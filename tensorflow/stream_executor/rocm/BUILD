# Description:
#   ROCm-platform specific StreamExecutor support code.

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "//tensorflow/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load("//tensorflow:tensorflow.bzl", "tf_copts")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")

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
    name = "rocm_platform_id",
    srcs = ["rocm_platform_id.cc"],
    hdrs = ["rocm_platform_id.h"],
    deps = ["//tensorflow/stream_executor:platform"],
)

cc_library(
    name = "rocm_platform",
    srcs = if_rocm_is_configured(["rocm_platform.cc"]),
    hdrs = if_rocm_is_configured(["rocm_platform.h"]),
    visibility = ["//visibility:public"],
    deps = if_rocm_is_configured([
        ":rocm_driver",
        ":rocm_gpu_executor",
        ":rocm_platform_id",
        "//tensorflow/stream_executor",  # buildcleaner: keep
        "//tensorflow/stream_executor:executor_cache",
        "//tensorflow/stream_executor:multi_platform_manager",
        "//tensorflow/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ]),
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "rocm_diagnostics",
    srcs = if_rocm_is_configured(["rocm_diagnostics.cc"]),
    hdrs = if_rocm_is_configured(["rocm_diagnostics.h"]),
    deps = if_rocm_is_configured([
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//tensorflow/stream_executor/gpu:gpu_diagnostics_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ]),
)

cc_library(
    name = "rocm_driver",
    srcs = if_rocm_is_configured(["rocm_driver.cc"]),
    hdrs = if_rocm_is_configured([
        "rocm_driver.h",
        "rocm_driver_wrapper.h",
    ]),
    deps = if_rocm_is_configured([
        ":rocm_diagnostics",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/stream_executor:device_options",
        "//tensorflow/stream_executor/gpu:gpu_driver_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "//tensorflow/stream_executor/platform:dso_loader",
    ]),
)

# The activation library is tightly coupled to the executor library.
# TODO(leary) split up rocm_gpu_executor.cc so that this can stand alone.
cc_library(
    name = "rocm_activation_header",
    hdrs = if_rocm_is_configured(["rocm_activation.h"]),
    visibility = ["//visibility:public"],
    deps = if_rocm_is_configured([
        "//tensorflow/stream_executor/gpu:gpu_activation_header",
        "//tensorflow/stream_executor/platform",
    ]),

)

cc_library(
    name = "rocm_activation",
    srcs = [],
    hdrs = if_rocm_is_configured(["rocm_activation.h"]),
    deps = if_rocm_is_configured([
        ":rocm_driver",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor:stream_executor_internal",
        "//tensorflow/stream_executor/gpu:gpu_activation",
        "//tensorflow/stream_executor/platform",
    ]),
)

cc_library(
    name = "rocm_gpu_executor_header",
    textual_hdrs = if_rocm_is_configured(["rocm_gpu_executor.h"]),
    visibility = ["//visibility:public"],
    deps = if_rocm_is_configured([
        ":rocm_kernel",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor/gpu:gpu_executor_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ]),
)

cc_library(
    name = "rocblas_plugin",
    srcs = if_rocm_is_configured(["rocm_blas.cc"]),
    hdrs = if_rocm_is_configured(["rocm_blas.h"]),
    visibility = ["//visibility:public"],
    deps = if_rocm_is_configured([
        ":rocm_activation",
        ":rocm_gpu_executor",
        ":rocm_helpers",
        ":rocm_platform_id",
        ":rocm_stream",
        ":rocm_timer",
        "@com_google_absl//absl/strings",
        "//third_party/eigen3",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/core:lib_internal",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:host_or_device_scalar",
        "//tensorflow/stream_executor:plugin_registry",
        "//tensorflow/stream_executor:scratch_allocator",
        "//tensorflow/stream_executor:timer",
        "//tensorflow/stream_executor/gpu:gpu_helpers_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "//tensorflow/stream_executor/platform:dso_loader",
    ] + if_static(["@local_config_rocm//rocm:rocblas"])),
    alwayslink = True,
)

cc_library(
    name = "rocfft_plugin",
    srcs = if_rocm_is_configured(["rocm_fft.cc"]),
    hdrs = if_rocm_is_configured(["rocm_fft.h"]),
    visibility = ["//visibility:public"],
    deps = if_rocm_is_configured([
        ":rocm_activation_header",
        ":rocm_gpu_executor_header",
        ":rocm_helpers",
        ":rocm_platform_id",
        ":rocm_stream",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:fft",
        "//tensorflow/stream_executor:plugin_registry",
        "//tensorflow/stream_executor:scratch_allocator",
        "//tensorflow/stream_executor/gpu:gpu_helpers_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "//tensorflow/stream_executor/platform:dso_loader",
    ] + if_static(["@local_config_rocm//rocm:rocfft"])),
    alwayslink = True,
)

cc_library(
    name = "miopen_plugin",
    srcs = if_rocm_is_configured(["rocm_dnn.cc"]),
    hdrs = if_rocm_is_configured(["rocm_dnn.h"]),
    visibility = ["//visibility:public"],
    deps = if_rocm_is_configured([
        ":rocm_activation",
        ":rocm_diagnostics",
        ":rocm_driver",
        ":rocm_gpu_executor",
        ":rocm_platform_id",
        ":rocm_stream",
        ":rocm_timer",
        "@local_config_rocm//rocm:rocm_headers",
        "@com_google_absl//absl/strings",
        "//third_party/eigen3",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/stream_executor:dnn",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:logging_proto_cc",
        "//tensorflow/stream_executor:plugin_registry",
        "//tensorflow/stream_executor:scratch_allocator",
        "//tensorflow/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/stream_executor:temporary_device_memory",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "//tensorflow/stream_executor/platform:dso_loader",
    ] + if_static(["@local_config_rocm//rocm:miopen"])),
    alwayslink = True,
)

cc_library(
    name = "rocrand_plugin",
    srcs = if_rocm_is_configured(["rocm_rng.cc"]),
    hdrs = if_rocm_is_configured(["rocm_rng.h"]),
    deps = if_rocm_is_configured([
        ":rocm_activation",
        ":rocm_gpu_executor",
        ":rocm_helpers",
        ":rocm_platform_id",
        ":rocm_stream",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:plugin_registry",
        "//tensorflow/stream_executor:rng",
        "//tensorflow/stream_executor/gpu:gpu_helpers_header",
        "//tensorflow/stream_executor/gpu:gpu_rng_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "//tensorflow/stream_executor/platform:dso_loader",
    ] + if_static(["@local_config_rocm//rocm:hiprand"])),
    alwayslink = True,
)

cc_library(
    name = "rocm_kernel",
    srcs = if_rocm_is_configured(["rocm_kernel.cc"]),
    hdrs = if_rocm_is_configured(["rocm_kernel.h"]),
    deps = if_rocm_is_configured([
        ":rocm_driver",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/stream_executor/gpu:gpu_kernel_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ]),
)

cc_library(
    name = "rocm_helpers",
    textual_hdrs = if_rocm_is_configured(["rocm_helpers.h"]),
    deps = if_rocm_is_configured([
        "//tensorflow/stream_executor/gpu:gpu_helpers_header",
    ]),
)

cc_library(
    name = "rocm_event",
    srcs = if_rocm_is_configured(["rocm_event.cc"]),
    hdrs = if_rocm_is_configured(["rocm_event.h"]),
    deps = if_rocm_is_configured([
        ":rocm_driver",
        ":rocm_gpu_executor_header",
        ":rocm_stream",
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor/gpu:gpu_event",
        "//tensorflow/stream_executor/gpu:gpu_stream_header",
        "//tensorflow/stream_executor/lib",
    ]),
)

cc_library(
    name = "rocm_stream",
    srcs = [],
    hdrs = if_rocm_is_configured(["rocm_stream.h"]),
    deps = if_rocm_is_configured([
        ":rocm_driver",
        ":rocm_gpu_executor_header",
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor:stream_header",
        "//tensorflow/stream_executor/gpu:gpu_stream",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ]),
)

cc_library(
    name = "rocm_timer",
    srcs = [],
    hdrs = if_rocm_is_configured(["rocm_timer.h"]),
    deps = if_rocm_is_configured([
        ":rocm_driver",
        ":rocm_gpu_executor_header",
        ":rocm_stream",
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor/gpu:gpu_timer",
        "//tensorflow/stream_executor/lib",
    ]),
)

cc_library(
    name = "rocm_gpu_executor",
    srcs = if_rocm_is_configured(["rocm_gpu_executor.cc"]),
    hdrs = if_rocm_is_configured(["rocm_gpu_executor.h"]),
    deps = if_rocm_is_configured([
        ":rocm_activation",
        ":rocm_diagnostics",
        ":rocm_driver",
        ":rocm_event",
        ":rocm_kernel",
        ":rocm_platform_id",
        ":rocm_stream",
        ":rocm_timer",
        "@com_google_absl//absl/strings",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:plugin_registry",
        "//tensorflow/stream_executor:stream_executor_internal",
        "//tensorflow/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/stream_executor:timer",
        "//tensorflow/stream_executor/gpu:gpu_executor_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "//tensorflow/stream_executor/platform:dso_loader",
    ]),
    alwayslink = True,
)

cc_library(
    name = "all_runtime",
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":miopen_plugin",
        ":rocblas_plugin",
        ":rocfft_plugin",
        ":rocm_driver",
        ":rocm_platform",
        ":rocrand_plugin",
    ],
    alwayslink = 1,
)
