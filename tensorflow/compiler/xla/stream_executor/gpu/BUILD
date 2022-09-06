# Description:
#   GPU-platform specific StreamExecutor support code.

load(
    "//tensorflow/compiler/xla/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load(
    "//tensorflow/tsl/platform/default:cuda_build_defs.bzl",
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
        "//tensorflow/compiler/xla/stream_executor:__subpackages__",
        "//tensorflow/core/common_runtime/gpu:__subpackages__",
        "//tensorflow/stream_executor:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "gpu_activation_header",
    hdrs = ["gpu_activation.h"],
    deps = ["//tensorflow/compiler/xla/stream_executor/platform"],
)

cc_library(
    name = "gpu_activation",
    srcs = if_gpu_is_configured(["gpu_activation.cc"]),
    hdrs = if_gpu_is_configured(["gpu_activation.h"]),
    deps = if_gpu_is_configured([
        ":gpu_executor_header",
        ":gpu_activation_header",
        ":gpu_driver_header",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_internal",
        "//tensorflow/compiler/xla/stream_executor/platform",
    ]),
)

cc_library(
    name = "gpu_diagnostics_header",
    hdrs = if_gpu_is_configured(["gpu_diagnostics.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/compiler/xla/stream_executor/platform",
    ],
)

cc_library(
    name = "gpu_driver_header",
    hdrs = if_gpu_is_configured(["gpu_driver.h"]),
    visibility = [
        "//tensorflow/compiler/xla/service/gpu:__subpackages__",
        "//tensorflow/compiler/xla/stream_executor:__subpackages__",
        "//tensorflow/core/common_runtime/gpu:__subpackages__",
        "//tensorflow/core/util/autotune_maps:__subpackages__",
        "//tensorflow/stream_executor:__subpackages__",
    ],
    deps = [
        ":gpu_types_header",
        "//tensorflow/compiler/xla/stream_executor:device_options",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/compiler/xla/stream_executor/platform",
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
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor/lib",
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
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/lib",
    ],
)

cc_library(
    name = "gpu_executor_header",
    hdrs = if_gpu_is_configured(["gpu_executor.h"]),
    deps = [
        ":gpu_kernel_header",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:platform",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/core:lib",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_helpers_header",
    hdrs = if_gpu_is_configured(["gpu_helpers.h"]),
    deps = [
        ":gpu_types_header",
        "//tensorflow/tsl/platform:logging",
    ],
)

cc_library(
    name = "gpu_kernel_header",
    hdrs = if_gpu_is_configured(["gpu_kernel.h"]),
    deps = [
        ":gpu_driver_header",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
    ],
)

cc_library(
    name = "gpu_rng_header",
    hdrs = if_gpu_is_configured(["gpu_rng.h"]),
    deps = [
        ":gpu_types_header",
        "//tensorflow/compiler/xla/stream_executor:plugin_registry",
        "//tensorflow/compiler/xla/stream_executor:rng",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "gpu_stream_header",
    hdrs = if_gpu_is_configured(["gpu_stream.h"]),
    deps = [
        ":gpu_driver_header",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_internal",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "gpu_stream",
    srcs = if_gpu_is_configured(["gpu_stream.cc"]),
    hdrs = if_gpu_is_configured(["gpu_stream.h"]),
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "gpu_timer_header",
    hdrs = if_gpu_is_configured(["gpu_timer.h"]),
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_internal",
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
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/lib",
    ],
)

cc_library(
    name = "gpu_types_header",
    hdrs = if_gpu_is_configured(["gpu_types.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/platform",
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
        "//tensorflow/compiler/xla/stream_executor:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
        "//tensorflow/stream_executor:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "asm_compiler_header",
    hdrs = if_gpu_is_configured(["asm_compiler.h"]),
    copts = tf_copts(),
    visibility = [
        "//tensorflow/compiler/mlir/tools/kernel_gen:__subpackages__",
        "//tensorflow/compiler/xla/service/gpu:__subpackages__",
        "//tensorflow/compiler/xla/stream_executor:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
        "//tensorflow/stream_executor:__subpackages__",
    ],
    deps = if_gpu_is_configured([
        ":gpu_asm_opts",
        ":gpu_driver_header",
        ":gpu_helpers_header",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/base:core_headers",
        "//tensorflow/tsl/platform:regexp",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:subprocess",
        "//tensorflow/core/platform:path",
        "//tensorflow/tsl/platform:cuda_libdevice_path",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ]) + if_cuda_is_configured([
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_driver",
    ]) + if_rocm_is_configured([
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_driver",
    ]),
)

cc_library(
    name = "asm_compiler",
    srcs = if_gpu_is_configured(["asm_compiler.cc"]),
    hdrs = if_gpu_is_configured(["asm_compiler.h"]),
    copts = tf_copts(),
    visibility = [
        "//tensorflow/compiler/mlir/tools/kernel_gen:__subpackages__",
        "//tensorflow/compiler/xla/service/gpu:__subpackages__",
        "//tensorflow/compiler/xla/stream_executor:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
        "//tensorflow/stream_executor:__subpackages__",
    ],
    deps = if_gpu_is_configured([
        ":gpu_asm_opts",
        ":gpu_driver_header",
        ":gpu_helpers_header",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/base:core_headers",
        "//tensorflow/tsl/platform:regexp",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:subprocess",
        "//tensorflow/core/platform:path",
        "//tensorflow/tsl/platform:cuda_libdevice_path",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ]) + if_cuda_is_configured([
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_asm_compiler",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_driver",
        "//tensorflow/compiler/xla/stream_executor/cuda:ptxas_wrapper",
        "//tensorflow/compiler/xla/stream_executor/cuda:fatbinary_wrapper",
    ]) + if_rocm_is_configured([
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_driver",
    ]),
)

cc_library(
    name = "redzone_allocator",
    srcs = if_gpu_is_configured(["redzone_allocator.cc"]),
    hdrs = if_gpu_is_configured(["redzone_allocator.h"]),
    copts = tf_copts(),
    visibility = [
        "//tensorflow/compiler/xla/service/gpu:__subpackages__",
        "//tensorflow/compiler/xla/stream_executor:__subpackages__",
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
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "//tensorflow/compiler/xla/stream_executor:device_memory_allocator",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
    ]),
)
