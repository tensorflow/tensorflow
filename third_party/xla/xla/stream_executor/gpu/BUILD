# Description:
#   GPU-platform specific StreamExecutor support code.

load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm",
    "if_rocm_is_configured",
)
load(
    "@local_tsl//tsl/platform:build_config_root.bzl",
    "if_static",
    "tf_gpu_tests_tags",
)
load(
    "@local_tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "//xla:xla.bzl",
    "xla_cc_test",
)
load(
    "//xla/service/gpu:build_defs.bzl",
    "gpu_kernel_library",
)
load(
    "//xla/stream_executor:build_defs.bzl",
    "gpu_only_cc_library",
    "if_gpu_is_configured",
)
load(
    "//xla/tests:build_defs.bzl",
    "xla_test",
)
load(
    "//xla/tsl:tsl.bzl",
    "if_libtpu",
    "internal_visibility",
    "tsl_copts",
    "tsl_gpu_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tensorflow/compiler/tf2xla:__subpackages__",
        "//xla:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
        "//xla/pjrt:__subpackages__",
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
        "//tensorflow/core/common_runtime/gpu:__subpackages__",
        "//waymo/ml/compiler/triton:__subpackages__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "gpu_activation_header",
    hdrs = ["gpu_activation.h"],
)

gpu_only_cc_library(
    name = "gpu_activation",
    srcs = ["gpu_activation.cc"],
    hdrs = ["gpu_activation.h"],
    deps = [
        ":gpu_activation_header",
        ":gpu_driver_header",
        ":gpu_executor_header",
        "//xla/stream_executor",
    ],
)

gpu_only_cc_library(
    name = "gpu_diagnostics_header",
    hdrs = ["gpu_diagnostics.h"],
    deps = ["@com_google_absl//absl/status:statusor"],
)

gpu_only_cc_library(
    name = "gpu_collectives_header",
    hdrs = ["gpu_collectives.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

gpu_only_cc_library(
    name = "gpu_driver_header",
    hdrs = ["gpu_driver.h"],
    visibility = internal_visibility([
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
        "//tensorflow/core/common_runtime/gpu:__subpackages__",
        "//tensorflow/core/util/autotune_maps:__subpackages__",
    ]),
    deps = [
        ":gpu_types_header",
        "//xla/stream_executor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ] + if_libtpu(
        if_false = ["@local_config_cuda//cuda:cuda_headers"],
        if_true = [],
    ),
)

gpu_only_cc_library(
    name = "gpu_runtime_header",
    hdrs = ["gpu_runtime.h"],
    visibility = internal_visibility([
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
    ]),
    deps = [
        ":gpu_types_header",
        "@com_google_absl//absl/status:statusor",
    ],
)

gpu_only_cc_library(
    name = "gpu_kernels",
    hdrs = ["gpu_kernels.h"],
)

gpu_only_cc_library(
    name = "gpu_command_buffer",
    srcs = ["gpu_command_buffer.cc"],
    hdrs = ["gpu_command_buffer.h"],
    local_defines = if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        ":gpu_kernel_header",
        ":gpu_kernels",
        ":gpu_stream",
        ":gpu_types_header",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_interface",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ] + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_conditional_kernels",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:hip_conditional_kernels",
    ]),
)

gpu_only_cc_library(
    name = "gpu_event_header",
    hdrs = ["gpu_event.h"],
    deps = [
        ":gpu_stream_header",
        ":gpu_types_header",
        "//xla/stream_executor:event_interface",
        "//xla/stream_executor:stream_executor_interface",
        "@com_google_absl//absl/status",
    ],
)

gpu_only_cc_library(
    name = "gpu_event",
    srcs = ["gpu_event.cc"],
    hdrs = ["gpu_event.h"],
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        ":gpu_stream",
        ":gpu_types_header",
        "//xla/stream_executor:event_interface",
        "//xla/stream_executor:stream_executor_interface",
        "@com_google_absl//absl/status",
    ],
)

gpu_only_cc_library(
    name = "gpu_executor_header",
    hdrs = ["gpu_executor.h"],
    deps = [
        ":gpu_collectives_header",
        ":gpu_driver_header",
        ":gpu_types_header",
        "//xla/stream_executor",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_interface",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

gpu_only_cc_library(
    name = "gpu_helpers_header",
    hdrs = ["gpu_helpers.h"],
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
    visibility = internal_visibility([
        "//xla/tsl:internal",
    ]),
    deps = [
        "@com_google_absl//absl/status",
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
    visibility = internal_visibility([
        "//tensorflow/compiler/tf2xla:__subpackages__",
        "//xla:__subpackages__",
        "//tensorflow/core/common_runtime/gpu:__subpackages__",
        "//tensorflow/stream_executor:__subpackages__",
    ]),
    deps = [
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:logging",
    ],
    alwayslink = True,
)

gpu_only_cc_library(
    name = "gpu_kernel_header",
    hdrs = ["gpu_kernel.h"],
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        ":gpu_types_header",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_interface",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:logging",
    ],
)

gpu_only_cc_library(
    name = "gpu_stream_header",
    hdrs = ["gpu_stream.h"],
    deps = [
        ":gpu_types_header",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_interface",
        "@com_google_absl//absl/log:check",
    ],
)

gpu_only_cc_library(
    name = "gpu_stream",
    srcs = ["gpu_stream.cc"],
    hdrs = ["gpu_stream.h"],
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        ":gpu_types_header",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_interface",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
    ],
)

gpu_only_cc_library(
    name = "gpu_timer_kernel_header",
    hdrs = ["gpu_timer_kernel.h"],
)

gpu_kernel_library(
    name = "gpu_timer_kernel",
    srcs = if_gpu_is_configured(["gpu_timer_kernel.cu.cc"]),
    deps = [
        ":gpu_timer_kernel_header",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

gpu_only_cc_library(
    name = "gpu_timer_header",
    hdrs = ["gpu_timer.h"],
    deps = [
        ":gpu_executor_header",
        ":gpu_timer_kernel_header",
        ":gpu_types_header",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)

gpu_only_cc_library(
    name = "gpu_timer",
    srcs = ["gpu_timer.cc"],
    hdrs = ["gpu_timer.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        ":gpu_stream",
        ":gpu_timer_kernel_header",
        ":gpu_types_header",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_interface",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/utility",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ] + if_gpu_is_configured([
        ":gpu_timer_kernel",
    ]) + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_driver",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:rocm_driver",
    ]),
)

gpu_only_cc_library(
    name = "gpu_types_header",
    hdrs = ["gpu_types.h"],
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
    visibility = internal_visibility([
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
    ]),
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

gpu_only_cc_library(
    name = "asm_compiler",
    srcs = ["asm_compiler.cc"],
    hdrs = ["asm_compiler.h"],
    copts = tsl_copts(),
    visibility = internal_visibility([
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
    ]),
    deps = [
        ":gpu_asm_opts",
        ":gpu_driver_header",
        ":gpu_helpers_header",
        ":gpu_types_header",
        "//xla:util",
        "//xla/stream_executor",
        "//xla/stream_executor/cuda:ptx_compiler",
        "//xla/stream_executor/cuda:ptx_compiler_support",
        "//xla/stream_executor/platform",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:cuda_libdevice_path",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:regexp",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:subprocess",
    ] + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_driver",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:rocm_driver",
    ]),
)

gpu_kernel_library(
    name = "redzone_allocator_kernel",
    srcs = if_gpu_is_configured(["redzone_allocator.cu.cc"]),
    hdrs = if_gpu_is_configured(["redzone_allocator.h"]),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = if_gpu_is_configured([
        ":gpu_asm_opts",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ]) + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

gpu_only_cc_library(
    name = "redzone_allocator",
    srcs = ["redzone_allocator.cc"],
    hdrs = ["redzone_allocator.h"],
    copts = tsl_copts(),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    visibility = internal_visibility([
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
    ]),
    deps = [
        ":asm_compiler",
        ":gpu_asm_opts",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/framework:allocator",
        "@local_tsl//tsl/lib/math:math_util",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ] + if_rocm_is_configured([
        ":redzone_allocator_kernel",
    ]) + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_asm_compiler",
    ]),
)

xla_cc_test(
    name = "redzone_allocator_test",
    srcs = if_gpu_is_configured(["redzone_allocator_test.cc"]),
    tags = tf_gpu_tests_tags() + [
        "no_cuda_asan",  # TODO(b/171512140): re-enable.
    ],
    deps = [
        ":gpu_asm_opts",
        ":gpu_init",
        ":redzone_allocator",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

# TODO(tlongeri): Remove gpu_cudamallocasync_allocator header/impl split
tsl_gpu_library(
    name = "gpu_cudamallocasync_allocator_header",
    hdrs = ["gpu_cudamallocasync_allocator.h"],
    deps = [
        "//xla/stream_executor:stream_executor_headers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_tsl//tsl/framework:allocator",
        "@local_tsl//tsl/framework:device_id",
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
    deps = [
        ":gpu_init_impl",
        "//xla/stream_executor:stream_executor_headers",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/framework:allocator",
        "@local_tsl//tsl/framework:device_id",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:mutex",
    ],
)

cc_library(
    name = "gpu_blas_lt",
    srcs = ["gpu_blas_lt.cc"],
    hdrs = ["gpu_blas_lt.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:algorithm_util",
        "//xla/stream_executor",
        "//xla/stream_executor:host_or_device_scalar",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/protobuf:dnn_proto_cc",
    ] + if_cuda_is_configured([
        "@local_tsl//tsl/platform:tensor_float_32_hdr_lib",
    ]) + if_static([
        "@local_tsl//tsl/platform:tensor_float_32_utils",
    ]),
)

gpu_kernel_library(
    name = "gpu_test_kernels",
    testonly = 1,
    srcs = if_gpu_is_configured(["gpu_test_kernels.cu.cc"]),
    hdrs = if_gpu_is_configured(["gpu_test_kernels.h"]),
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "//xla/stream_executor/rocm:add_i32_kernel",
    ]),
)

xla_test(
    name = "gpu_kernel_test",
    srcs = if_gpu_is_configured(["gpu_kernel_test.cc"]),
    backends = ["gpu"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured(["TENSORFLOW_USE_ROCM=1"]),
    deps = [
        ":gpu_test_kernels",
        "//xla/service:platform_util",
        "//xla/stream_executor",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ] + if_cuda([
        "//xla/stream_executor/cuda:cuda_platform",
    ]) + if_rocm([
        "//xla/stream_executor/rocm:rocm_platform",
    ]),
)

xla_test(
    name = "gpu_command_buffer_test",
    srcs = if_gpu_is_configured(["gpu_command_buffer_test.cc"]),
    backends = ["gpu"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured(["TENSORFLOW_USE_ROCM=1"]),
    deps = [
        ":gpu_command_buffer",
        ":gpu_test_kernels",
        ":gpu_types_header",
        "//xla/service:platform_util",
        "//xla/stream_executor",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ] + if_cuda([
        "//xla/stream_executor/cuda:cuda_platform",
    ]) + if_rocm([
        "//xla/stream_executor/rocm:rocm_platform",
    ]),
)

xla_cc_test(
    name = "memcpy_test",
    srcs = ["memcpy_test.cc"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    tags = tf_gpu_tests_tags(),
    deps = [
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:platform_manager",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ] + if_cuda([
        "//xla/stream_executor/cuda:cuda_platform",
    ]) + if_rocm([
        "//xla/stream_executor/rocm:rocm_platform",
    ]),
)

xla_cc_test(
    name = "stream_search_test",
    size = "small",
    srcs = ["stream_search_test.cc"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    tags = tf_gpu_tests_tags() + [
        "requires-gpu-nvidia",
    ],
    deps = [
        "//xla/stream_executor",
        "//xla/stream_executor/host:host_platform",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ] + if_cuda([
        "//xla/stream_executor/cuda:cuda_platform",
    ]) + if_rocm([
        "//xla/stream_executor/rocm:rocm_platform",
    ]),
)
