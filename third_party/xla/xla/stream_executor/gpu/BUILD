# Description:
#   GPU-platform specific StreamExecutor support code.

load(
    "//xla/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@local_tsl//tsl:tsl.bzl",
    "if_libtpu",
    "set_external_visibility",
    "tsl_copts",
    "tsl_gpu_library",
)
load(
    "@local_tsl//tsl/platform:build_config_root.bzl",
    "if_static",
)
load(
    "@local_tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "gpu_activation_header",
    hdrs = ["gpu_activation.h"],
    visibility = ["//visibility:public"],
    deps = ["//xla/stream_executor/platform"],
)

cc_library(
    name = "gpu_activation",
    srcs = if_gpu_is_configured(["gpu_activation.cc"]),
    hdrs = if_gpu_is_configured(["gpu_activation.h"]),
    visibility = ["//visibility:public"],
    deps = if_gpu_is_configured([
        ":gpu_executor_header",
        ":gpu_activation_header",
        ":gpu_driver_header",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_internal",
        "//xla/stream_executor/platform",
    ]),
)

cc_library(
    name = "gpu_diagnostics_header",
    hdrs = if_gpu_is_configured(["gpu_diagnostics.h"]),
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor/platform",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "gpu_driver_header",
    hdrs = if_gpu_is_configured(["gpu_driver.h"]),
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_types_header",
        "//xla/stream_executor:device_options",
        "//xla/stream_executor:stream_executor_headers",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ] + if_libtpu(
        if_false = ["@local_config_cuda//cuda:cuda_headers"],
        if_true = [],
    ),
)

cc_library(
    name = "gpu_command_buffer",
    srcs = if_gpu_is_configured(["gpu_command_buffer.cc"]),
    hdrs = if_gpu_is_configured(["gpu_command_buffer.h"]),
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        ":gpu_kernel_header",
        ":gpu_stream",
        ":gpu_types_header",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor:stream_executor_internal",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "gpu_event_header",
    hdrs = if_gpu_is_configured(["gpu_event.h"]),
    visibility = ["//visibility:public"],
    deps = if_gpu_is_configured([
        ":gpu_driver_header",
        ":gpu_stream_header",
        "//xla/stream_executor:stream_executor_headers",
        "@local_tsl//tsl/platform:status",
    ]),
)

cc_library(
    name = "gpu_event",
    srcs = if_gpu_is_configured(["gpu_event.cc"]),
    hdrs = if_gpu_is_configured(["gpu_event.h"]),
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        ":gpu_stream",
        "//xla/stream_executor:stream_executor_headers",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "gpu_executor_header",
    hdrs = if_gpu_is_configured(["gpu_executor.h"]),
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_kernel_header",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor:stream_executor_internal",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "gpu_helpers_header",
    hdrs = if_gpu_is_configured(["gpu_helpers.h"]),
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_types_header",
        "@local_tsl//tsl/platform:logging",
    ],
)

tsl_gpu_library(
    name = "gpu_init",
    hdrs = [
        "gpu_init.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@local_tsl//tsl/platform:status",
    ] + if_static(
        [":gpu_init_impl"],
    ),
)

tsl_gpu_library(
    name = "gpu_init_impl",
    srcs = [
        "gpu_init.cc",
    ],
    hdrs = [
        "gpu_init.h",
    ],
    copts = tsl_copts(),
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor:platform",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "gpu_kernel_header",
    hdrs = if_gpu_is_configured(["gpu_kernel.h"]),
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_driver_header",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor:stream_executor_internal",
        "//xla/stream_executor/platform",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "gpu_stream_header",
    hdrs = if_gpu_is_configured(["gpu_stream.h"]),
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_types_header",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor:stream_executor_internal",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "gpu_stream",
    srcs = if_gpu_is_configured(["gpu_stream.cc"]),
    hdrs = if_gpu_is_configured(["gpu_stream.h"]),
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_executor_header",
        ":gpu_types_header",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor:stream_executor_internal",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "gpu_timer_header",
    hdrs = if_gpu_is_configured(["gpu_timer.h"]),
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor:stream_executor_internal",
    ],
)

cc_library(
    name = "gpu_timer",
    srcs = if_gpu_is_configured(["gpu_timer.cc"]),
    hdrs = if_gpu_is_configured(["gpu_timer.h"]),
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        ":gpu_stream",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor:stream_executor_internal",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:status",
    ] + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_driver",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:rocm_driver",
    ]),
)

cc_library(
    name = "gpu_types_header",
    hdrs = if_gpu_is_configured(["gpu_types.h"]),
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor/platform",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

cc_library(
    name = "gpu_asm_opts",
    hdrs = ["gpu_asm_opts.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "asm_compiler_header",
    hdrs = if_gpu_is_configured(["asm_compiler.h"]),
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = if_gpu_is_configured([
        ":gpu_asm_opts",
        ":gpu_driver_header",
        ":gpu_helpers_header",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/base:core_headers",
        "@local_tsl//tsl/platform:regexp",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:subprocess",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:cuda_libdevice_path",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ]) + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_driver",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:rocm_driver",
    ]) + ["@local_tsl//tsl/platform:statusor"],
)

cc_library(
    name = "asm_compiler",
    srcs = if_gpu_is_configured(["asm_compiler.cc"]),
    hdrs = if_gpu_is_configured(["asm_compiler.h"]),
    copts = tsl_copts(),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    visibility = ["//visibility:public"],
    deps = if_gpu_is_configured([
        ":gpu_asm_opts",
        ":gpu_driver_header",
        ":gpu_helpers_header",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/base:core_headers",
        "@local_tsl//tsl/platform:regexp",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:subprocess",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:cuda_libdevice_path",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/stream_executor/platform",
        "//xla:util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ]) + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_asm_compiler",
        "//xla/stream_executor/cuda:cuda_driver",
        "//xla/stream_executor/cuda:ptxas_wrapper",
        "//xla/stream_executor/cuda:nvlink_wrapper",
        "//xla/stream_executor/cuda:fatbinary_wrapper",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:rocm_driver",
    ]) + ["@local_tsl//tsl/platform:statusor"],
)

cc_library(
    name = "redzone_allocator",
    srcs = if_gpu_is_configured(["redzone_allocator.cc"]),
    hdrs = if_gpu_is_configured(["redzone_allocator.h"]),
    copts = tsl_copts(),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    visibility = ["//visibility:public"],
    deps = if_gpu_is_configured([
        ":asm_compiler",
        ":gpu_asm_opts",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@local_tsl//tsl/lib/math:math_util",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/framework:allocator",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:stream_executor_headers",
        "@local_tsl//tsl/platform:status",
    ]),
)

# TODO(tlongeri): Remove gpu_cudamallocasync_allocator header/impl split
tsl_gpu_library(
    name = "gpu_cudamallocasync_allocator_header",
    hdrs = ["gpu_cudamallocasync_allocator.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_tsl//tsl/framework:allocator",
        "@local_tsl//tsl/framework:device_id",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:mutex",
    ],
)

tsl_gpu_library(
    name = "gpu_cudamallocasync_allocator",
    srcs = [
        "gpu_cudamallocasync_allocator.cc",
    ],
    hdrs = ["gpu_cudamallocasync_allocator.h"],
    cuda_deps = [
        "//xla/stream_executor/cuda:cuda_activation",
        "//xla/stream_executor/cuda:cuda_executor",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_init_impl",
        "//xla/stream_executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/framework:allocator",
        "@local_tsl//tsl/framework:device_id",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/util:env_var",
    ],
)

cc_library(
    name = "gpu_graph",
    srcs = if_gpu_is_configured(["gpu_graph.cc"]),
    hdrs = if_gpu_is_configured(["gpu_graph.h"]),
    visibility = ["//visibility:public"],
    deps = if_gpu_is_configured([
        ":gpu_driver_header",
        ":gpu_kernel_header",
        ":gpu_types_header",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/functional:any_invocable",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/stream_executor",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ]) + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//xla/stream_executor/cuda:cuda_driver",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:rocm_driver",
    ]) + ["@com_google_absl//absl/strings"],
)

cc_library(
    name = "gpu_blas_lt",
    srcs = ["gpu_blas_lt.cc"],
    hdrs = ["gpu_blas_lt.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:host_or_device_scalar",
        "//xla/stream_executor:stream_executor_headers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
    ] + if_cuda_is_configured([
        "@local_tsl//tsl/platform:tensor_float_32_hdr_lib",
    ]) + if_static([
        "@local_tsl//tsl/platform:tensor_float_32_utils",
    ]),
)
