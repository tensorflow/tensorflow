# Description:
#   GPU-platform specific StreamExecutor support code.

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("//xla:xla.default.bzl", "xla_cc_test")
load(
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_gpu_friends",
)
load(
    "//xla/tests:build_defs.bzl",
    "xla_test",
)
load(
    "//xla/tsl:tsl.bzl",
    "internal_visibility",
    "tsl_copts",
)
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "if_static",
)
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_gpu_friends(),
)

cc_library(
    name = "context",
    hdrs = ["context.h"],
    deps = [
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "context_map",
    hdrs = ["context_map.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_cc_test(
    name = "context_map_test",
    srcs = ["context_map_test.cc"],
    deps = [
        ":context_map",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "mock_context",
    testonly = True,
    hdrs = ["mock_context.h"],
    deps = [
        ":context",
        "//xla/hlo/testlib:test",
    ],
)

cc_library(
    name = "read_numa_node",
    srcs = ["read_numa_node.cc"],
    hdrs = ["read_numa_node.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "scoped_activate_context",
    srcs = ["scoped_activate_context.cc"],
    hdrs = ["scoped_activate_context.h"],
    deps = [
        ":context",
        "//xla/stream_executor:activate_context",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "scoped_activate_context_test",
    srcs = ["scoped_activate_context_test.cc"],
    deps = [
        ":mock_context",
        ":scoped_activate_context",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "gpu_command_buffer",
    srcs = ["gpu_command_buffer.cc"],
    hdrs = ["gpu_command_buffer.h"],
    deps = [
        "//xla:debug_options_flags",
        "//xla/service:dump",
        "//xla/stream_executor:bit_pattern",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gpu_executor_header",
    hdrs = ["gpu_executor.h"],
    deps = [
        ":multicast_memory",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_common",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "multicast_memory",
    hdrs = ["multicast_memory.h"],
    deps = [
        "//xla/stream_executor:device_address",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "gpu_helpers_header",
    hdrs = ["gpu_helpers.h"],
    deps = [
        "//xla/stream_executor:device_address",
    ],
)

cc_library(
    name = "gpu_init",
    hdrs = [
        "gpu_init.h",
    ],
    visibility = internal_visibility([
        "//xla/tsl:internal",
    ]),
    deps = [
        "@com_google_absl//absl/status",
        "@tsl//tsl/platform:status",
    ] + if_static(
        [":gpu_init_impl"],
    ),
)

cc_library(
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
        "@tsl//tsl/platform:logging",
    ],
    alwayslink = True,
)

cc_library(
    name = "gpu_semaphore",
    srcs = ["gpu_semaphore.cc"],
    hdrs = ["gpu_semaphore.h"],
    deps = [
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:statusor",
    ],
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

cc_library(
    name = "asm_compiler",
    srcs = ["asm_compiler.cc"],
    hdrs = ["asm_compiler.h"],
    copts = tsl_copts(),
    visibility = internal_visibility([
        "//tensorflow/compiler/mlir/tools/kernel_gen:__subpackages__",
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
        "//tensorflow/compiler/mlir/tools/kernel_gen:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
    ]),
    deps = [
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:subprocess",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "redzone_allocator_kernel",
    hdrs = [
        "redzone_allocator_kernel.h",
    ],
    deps = [
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
    ],
)

cc_library(
    name = "redzone_allocator",
    srcs = [
        "redzone_allocator.cc",
    ],
    hdrs = ["redzone_allocator.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":gpu_kernel_registry",
        ":redzone_allocator_kernel",
        "//xla:shape_util",
        "//xla/service/gpu:stream_executor_util",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:device_address_handle",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/framework:allocator",
        "//xla/tsl/lib/math:math_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_test(
    name = "redzone_allocator_test",
    srcs = ["redzone_allocator_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_init",
        ":redzone_allocator",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "gpu_cudamallocasync_allocator",
    srcs = ["gpu_cudamallocasync_allocator.cc"],
    hdrs = ["gpu_cudamallocasync_allocator.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":gpu_init_impl",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_executor",
        "//xla/stream_executor/cuda:cuda_status",
        "//xla/tsl/framework:allocator",
        "//xla/tsl/framework:device_id",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

xla_test(
    name = "gpu_cudamallocasync_allocator_test",
    srcs = ["gpu_cudamallocasync_allocator_test.cc"],
    backends = ["nvgpu_any"],
    tags = ["cuda-only"],
    deps = [
        ":gpu_cudamallocasync_allocator",
        "//xla/service:platform_util",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_platform",
        "//xla/tsl/framework:device_id",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "gpu_blas_lt",
    srcs = ["gpu_blas_lt.cc"],
    hdrs = ["gpu_blas_lt.h"],
    deps = [
        ":gpu_blas_lt_proto_cc",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:algorithm_util",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:tensor_float_32_hdr_lib",
    ] + if_static([
        "@tsl//tsl/platform:tensor_float_32_utils",
    ]),
)

tf_proto_library(
    name = "gpu_blas_lt_proto",
    srcs = ["gpu_blas_lt.proto"],
    deps = [
        "//xla:xla_data_proto",
        "//xla/stream_executor:blas_proto",
    ],
)

xla_cc_test(
    name = "gpu_blas_lt_test",
    srcs = ["gpu_blas_lt_test.cc"],
    deps = [
        ":gpu_blas_lt",
        ":gpu_blas_lt_proto_cc",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:blas",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "gpu_test_kernel_traits",
    hdrs = ["gpu_test_kernel_traits.h"],
    deps = [
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
    ],
)

cc_library(
    name = "gpu_test_kernels",
    testonly = 1,
    srcs = ["gpu_test_kernels.cc"],
    hdrs = ["gpu_test_kernels.h"],
    linkstatic = True,
    tags = [
        "gpu",
        "no-oneapi",
    ],
    deps = [
        ":gpu_kernel_registry",
        ":gpu_test_kernel_traits",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "gpu_test_kernels_fatbin",
    testonly = True,
    srcs = ["gpu_test_kernels_fatbin.cc"],
    hdrs = ["gpu_test_kernels_fatbin.h"],
    data = if_cuda_is_configured([
        "//xla/stream_executor/cuda:gpu_test_kernels_cuda",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:gpu_test_kernels_rocm",
    ]),
    tags = ["gpu"],
    deps = [
        ":gpu_init_impl",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "gpu_test_kernels_fatbin_test",
    srcs = ["gpu_test_kernels_fatbin_test.cc"],
    tags = ["gpu"],
    deps = [
        ":gpu_test_kernels_fatbin",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "gpu_kernel_test",
    srcs = ["gpu_kernel_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_test_kernels",
        ":gpu_test_kernels_fatbin",
        ":tma_metadata",
        ":tma_metadata_proto_cc",
        "//xla/service:platform_util",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_args",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:typed_kernel_factory",
        "//xla/stream_executor/rocm:rocm_platform_id",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util/proto:parse_text_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "gpu_command_buffer_test",
    srcs = ["gpu_command_buffer_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_test_kernels",
        "//xla/service:platform_util",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:trace_command_buffer_factory",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "memcpy_test",
    srcs = ["memcpy_test.cc"],
    backends = ["gpu"],
    deps = [
        "//xla/service:platform_util",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "stream_search_test",
    size = "small",
    srcs = ["stream_search_test.cc"],
    backends = ["gpu"],
    deps = [
        "//xla/service:platform_util",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_finder",
        "//xla/stream_executor/host:host_platform",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "gpu_executor_test",
    size = "small",
    srcs = ["gpu_executor_test.cc"],
    backends = ["gpu"],
    deps = [
        "//xla/service:platform_util",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_space",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_test(
    name = "gpu_device_info_test",
    srcs = ["gpu_device_info_test.cc"],
    backends = ["gpu"],
    data = ["//xla/backends/gpu/target_config:all_gpu_specs"],
    deps = [
        "//xla/service:platform_util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

tf_proto_library(
    name = "tma_metadata_proto",
    srcs = ["tma_metadata.proto"],
)

cc_library(
    name = "tma_metadata",
    srcs = ["tma_metadata.cc"],
    hdrs = ["tma_metadata.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":tma_metadata_proto_cc",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "tma_metadata_test",
    srcs = ["tma_metadata_test.cc"],
    deps = [
        ":tma_metadata",
        ":tma_metadata_proto_cc",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "gpu_kernel_registry",
    srcs = ["gpu_kernel_registry.cc"],
    hdrs = ["gpu_kernel_registry.h"],
    deps = [
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:typed_kernel_factory",
        "//xla/stream_executor/platform:initialize",
        "//xla/stream_executor/platform:platform_object_registry",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_cc_test(
    name = "gpu_kernel_registry_test",
    srcs = ["gpu_kernel_registry_test.cc"],
    deps = [
        ":gpu_kernel_registry",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/platform:platform_object_registry",
        "//xla/stream_executor/rocm:rocm_platform_id",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "buffer_comparator_kernel",
    hdrs = ["buffer_comparator_kernel.h"],
    deps = [
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
    ],
)

exports_files([
    "buffer_comparator_kernel_lib.cu.h",
    "all_reduce_kernel_lib.cu.h",
    "ragged_all_to_all_kernel_lib.cu.h",
    "repeat_buffer_kernel.cu.h",
    "redzone_allocator_kernel_lib.cu.h",
    "gpu_test_kernels_lib.cu.h",
])

cc_library(
    name = "make_batch_pointers_kernel",
    hdrs = ["make_batch_pointers_kernel.h"],
    deps = [
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
    ],
)

cc_library(
    name = "ragged_all_to_all_kernel",
    hdrs = ["ragged_all_to_all_kernel.h"],
    deps = [
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
    ],
)

cc_library(
    name = "all_reduce_kernel",
    hdrs = ["all_reduce_kernel.h"],
    deps = [
        ":collective_kernel_metadata",
        "//xla/service:collective_ops_utils",
        "//xla/stream_executor:kernel",
    ],
)

cc_library(
    name = "collective_kernel_metadata",
    hdrs = ["collective_kernel_metadata.h"],
)

cc_library(
    name = "topk_kernel",
    hdrs = ["topk_kernel.h"],
    deps = [
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
    ],
)

cc_library(
    name = "repeat_buffer_kernel",
    hdrs = ["repeat_buffer_kernel.h"],
    deps = [
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
    ],
)

xla_test(
    name = "repeat_buffer_kernel_test",
    srcs = ["repeat_buffer_kernel_test.cc"],
    backends = ["gpu"],
    disabled_backends = [],
    deps = [
        ":gpu_kernel_registry",
        ":repeat_buffer_kernel",
        "//xla/service:platform_util",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "elf_section_extraction",
    srcs = ["elf_section_extraction.cc"],
    hdrs = ["elf_section_extraction.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "elf_section_extraction_test",
    srcs = ["elf_section_extraction_test.cc"],
    deps = [
        ":elf_section_extraction",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "buffer_debug_log",
    srcs = ["buffer_debug_log.cc"],
    hdrs = ["buffer_debug_log.h"],
    deps = [
        "//xla/backends/gpu/runtime:buffer_debug_log_proto_cc",
        "//xla/backends/gpu/runtime:buffer_debug_log_structs",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:stream",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_test(
    name = "buffer_debug_log_test",
    srcs = ["buffer_debug_log_test.cc"],
    backends = ["gpu"],
    deps = [
        ":buffer_debug_log",
        "//xla/backends/gpu/runtime:buffer_debug_log_proto_cc",
        "//xla/backends/gpu/runtime:buffer_debug_log_structs",
        "//xla/backends/gpu/runtime:thunk_id",
        "//xla/service:platform_util",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "buffer_debug_xor_checksum_kernel",
    hdrs = ["buffer_debug_xor_checksum_kernel.h"],
    deps = [
        "//xla/backends/gpu/runtime:buffer_debug_log_structs",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
    ],
)

cc_library(
    name = "buffer_debug_float_check_kernel",
    hdrs = ["buffer_debug_float_check_kernel.h"],
    deps = [
        "//xla:types",
        "//xla/backends/gpu/runtime:buffer_debug_log_structs",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
    ],
)

cc_library(
    name = "prefix_sum_kernel",
    hdrs = ["prefix_sum_kernel.h"],
    deps = [
        "//xla:types",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
    ],
)

cc_library(
    name = "kernel_serialization_check",
    testonly = True,
    srcs = ["kernel_serialization_check.cc"],
    hdrs = ["kernel_serialization_check.h"],
    deps = [
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:kernel_symbol_registry",
        "//xla/stream_executor:platform",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_for_library",
    ],
)
