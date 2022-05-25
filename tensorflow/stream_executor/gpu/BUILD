# Description:
#   GPU-platform specific StreamExecutor support code.

load("//tensorflow:tensorflow.bzl", "filegroup")
load(
    "//tensorflow/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load(
    "//tensorflow/core/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("//tensorflow:tensorflow.bzl", "if_libtpu", "tf_copts")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        "//tensorflow/compiler/tf2xla:__subpackages__",
        "//tensorflow/compiler/xla/service/gpu:__subpackages__",
        "//tensorflow/stream_executor:__subpackages__",
    ],
    licenses = ["notice"],
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
        ":gpu_executor_header",
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
    visibility = [
        "//tensorflow/compiler/xla/service/gpu:__subpackages__",
        "//tensorflow/core/util/autotune_maps:__subpackages__",
        "//tensorflow/stream_executor:__subpackages__",
    ],
    deps = [
        ":gpu_types_header",
        "//tensorflow/stream_executor:device_options",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ] + if_libtpu(
        if_false = ["@local_config_cuda//cuda:cuda_headers"],
        if_true = [],
    ),
)

cc_library(
    name = "gpu_event_header",
    hdrs = if_gpu_is_configured(["gpu_event.h"]),
    deps = if_gpu_is_configured([
        ":gpu_driver_header",
        ":gpu_stream_header",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor/lib",
    ]),
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
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:platform",
        "//tensorflow/stream_executor:stream_executor_internal",
        "//tensorflow/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "gpu_helpers_header",
    hdrs = if_gpu_is_configured(["gpu_helpers.h"]),
    deps = [
        ":gpu_types_header",
        "//tensorflow/core/platform:logging",
    ],
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
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor:plugin_registry",
        "//tensorflow/stream_executor:rng",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "gpu_stream_header",
    hdrs = if_gpu_is_configured(["gpu_stream.h"]),
    deps = [
        ":gpu_driver_header",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor:stream_executor_internal",
    ],
)

cc_library(
    name = "gpu_stream",
    srcs = if_gpu_is_configured(["gpu_stream.cc"]),
    hdrs = if_gpu_is_configured(["gpu_stream.h"]),
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        "//tensorflow/core:lib",
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

cc_library(
    name = "gpu_asm_opts",
    hdrs = ["gpu_asm_opts.h"],
    visibility = [
        "//tensorflow/compiler/xla/service/gpu:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
        "//tensorflow/stream_executor:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "asm_compiler",
    srcs = if_gpu_is_configured(["asm_compiler.cc"]),
    hdrs = if_gpu_is_configured(["asm_compiler.h"]),
    copts = tf_copts(),
    visibility = [
        "//tensorflow/compiler/mlir/tools/kernel_gen:__subpackages__",
        "//tensorflow/compiler/xla/service/gpu:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
        "//tensorflow/stream_executor:__subpackages__",
    ],
    deps = if_gpu_is_configured([
        ":gpu_asm_opts",
        ":gpu_driver_header",
        ":gpu_helpers_header",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/platform:cuda_libdevice_path",
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/container:flat_hash_map",
    ]) + if_cuda_is_configured([
        "//tensorflow/stream_executor/cuda:cuda_driver",
        "//tensorflow/stream_executor/cuda:ptxas_wrapper",
        "//tensorflow/stream_executor/cuda:fatbinary_wrapper",
    ]) + if_rocm_is_configured([
        "//tensorflow/stream_executor/rocm:rocm_driver",
    ]) + ["@com_google_absl//absl/container:flat_hash_set"],
)

cc_library(
    name = "redzone_allocator",
    srcs = if_gpu_is_configured(["redzone_allocator.cc"]),
    hdrs = if_gpu_is_configured(["redzone_allocator.h"]),
    copts = tf_copts(),
    visibility = [
        "//tensorflow/compiler/xla/service/gpu:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
        "//tensorflow/stream_executor:__subpackages__",
    ],
    deps = if_gpu_is_configured([
        ":asm_compiler",
        ":gpu_asm_opts",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:device_memory_allocator",
        "//tensorflow/stream_executor:stream_executor_headers",
    ]),
)
