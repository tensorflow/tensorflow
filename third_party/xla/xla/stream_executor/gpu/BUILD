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
    "@local_config_sycl//sycl:build_defs.bzl",
    "if_sycl_is_configured",
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
load("//xla:xla.bzl", "xla_cc_test")
load(
    "//xla/service/gpu:build_defs.bzl",
    "gpu_kernel_library",
)
load(
    "//xla/stream_executor:build_defs.bzl",
    "gpu_only_cc_library",
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
        "//learning/dist_belief_extra/gpu:__subpackages__",
        "//tensorflow/compiler/tf2xla:__subpackages__",
        "//xla:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
        "//xla/pjrt:__subpackages__",
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
        "//tensorflow/core:__subpackages__",
        "//third_party/sparse_conv:__subpackages__",
        "//waymo/ml/compiler/triton:__subpackages__",
        "//waymo/research/representation/threed:__subpackages__",
    ]),
    licenses = ["notice"],
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
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "mock_context",
    testonly = True,
    hdrs = ["mock_context.h"],
    deps = [
        ":context",
        "//xla:test",
    ],
)

cc_library(
    name = "read_numa_node",
    srcs = ["read_numa_node.cc"],
    hdrs = ["read_numa_node.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "scoped_activate_context",
    srcs = ["scoped_activate_context.cc"],
    hdrs = ["scoped_activate_context.h"],
    deps = [
        ":context",
        ":gpu_executor_header",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/log:check",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "scoped_activate_context_test",
    srcs = ["scoped_activate_context_test.cc"],
    deps = [
        ":mock_context",
        ":scoped_activate_context",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

gpu_only_cc_library(
    name = "gpu_diagnostics_header",
    hdrs = ["gpu_diagnostics.h"],
    deps = ["@com_google_absl//absl/status:statusor"],
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
        ":context",
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
        ":gpu_stream",
        ":gpu_types_header",
        "//xla:util",
        "//xla/stream_executor",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:typed_kernel_factory",
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
        "@local_config_cuda//cuda:cuda_headers",
        "//xla/stream_executor/cuda:command_buffer_kernels",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:command_buffer_kernels",
    ]),
)

cc_library(
    name = "gpu_executor_header",
    hdrs = ["gpu_executor.h"],
    deps = [
        ":context",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:host_memory_allocation",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_common",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "mock_gpu_executor",
    testonly = True,
    hdrs = ["mock_gpu_executor.h"],
    tags = ["gpu"],
    deps = [
        ":gpu_executor_header",
        "//xla:test",
        "//xla/stream_executor:allocator_stats",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:event",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:module_spec",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

gpu_only_cc_library(
    name = "gpu_helpers_header",
    hdrs = ["gpu_helpers.h"],
    deps = [
        "//xla/stream_executor:device_memory",
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
        ":context",
        ":gpu_executor_header",
        ":gpu_types_header",
        "//xla/stream_executor",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:logging",
    ],
)

gpu_only_cc_library(
    name = "gpu_stream_header",
    hdrs = ["gpu_stream.h"],
    deps = [
        ":gpu_executor_header",
        ":gpu_types_header",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:event",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_common",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
    ],
)

gpu_only_cc_library(
    name = "gpu_stream",
    srcs = ["gpu_stream.cc"],
    hdrs = ["gpu_stream.h"],
    deps = [
        ":gpu_driver_header",
        ":gpu_executor_header",
        ":gpu_kernel_header",
        ":gpu_types_header",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:event",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_common",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:nvtx_utils",
    ],
)

gpu_only_cc_library(
    name = "gpu_semaphore",
    srcs = ["gpu_semaphore.cc"],
    hdrs = ["gpu_semaphore.h"],
    deps = [
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
    ],
)

gpu_only_cc_library(
    name = "gpu_types_header",
    hdrs = ["gpu_types.h"],
    defines = if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]) + if_sycl_is_configured([
        "TENSORFLOW_USE_SYCL=1",
    ]),
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]) + if_sycl_is_configured([
        "@local_config_sycl//sycl:sycl_headers",
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
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:subprocess",
    ],
)

cc_library(
    name = "redzone_allocator_kernel_cuda",
    srcs = [
        "redzone_allocator_kernel.h",
        "redzone_allocator_kernel_cuda.cc",
    ],
    tags = [
        "cuda-only",
        "gpu",
        "manual",
    ],
    deps = [
        ":gpu_asm_opts",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:typed_kernel_factory",
        "//xla/stream_executor/cuda:cuda_asm_compiler",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

gpu_kernel_library(
    name = "redzone_allocator_kernel_rocm",
    srcs = [
        "redzone_allocator_kernel.h",
        "redzone_allocator_kernel_rocm.cu.cc",
    ],
    tags = ["manual"],
    deps = [
        ":gpu_asm_opts",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:typed_kernel_factory",
        "@com_google_absl//absl/status:statusor",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

gpu_only_cc_library(
    name = "redzone_allocator",
    srcs = [
        "redzone_allocator.cc",
        "redzone_allocator_kernel.h",
    ],
    hdrs = ["redzone_allocator.h"],
    visibility = internal_visibility([
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
    ]),
    deps = [
        ":gpu_asm_opts",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_handle",
        "//xla/stream_executor:scratch_allocator",
        "//xla/tsl/framework:allocator",
        "//xla/tsl/lib/math:math_util",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ] + if_rocm_is_configured([
        ":redzone_allocator_kernel_rocm",
    ]) + if_cuda_is_configured([
        ":redzone_allocator_kernel_cuda",
    ]),
)

xla_test(
    name = "redzone_allocator_test",
    srcs = ["redzone_allocator_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_asm_opts",
        ":gpu_init",
        ":redzone_allocator",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
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
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_executor",
        "//xla/stream_executor/cuda:cuda_status",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/tsl/framework:allocator",
        "//xla/tsl/framework:device_id",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_test(
    name = "gpu_cudamallocasync_allocator_test",
    srcs = ["gpu_cudamallocasync_allocator_test.cc"],
    backends = ["gpu_any"],
    tags = ["cuda-only"],
    deps = [
        ":gpu_cudamallocasync_allocator",
        ":gpu_stream",
        "//xla/service:platform_util",
        "//xla/stream_executor",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor/cuda:cuda_platform",
        "//xla/tsl/framework:device_id",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
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
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:algorithm_util",
        "//xla/stream_executor",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:host_or_device_scalar",
        "//xla/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ] + if_cuda_is_configured([
        "@local_tsl//tsl/platform:tensor_float_32_hdr_lib",
    ]) + if_static([
        "@local_tsl//tsl/platform:tensor_float_32_utils",
    ]),
)

gpu_kernel_library(
    name = "gpu_test_kernels",
    testonly = 1,
    srcs = ["gpu_test_kernels.cu.cc"],
    hdrs = ["gpu_test_kernels.h"],
    tags = ["gpu"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

genrule(
    name = "gpu_test_kernels_fatbin_extractor",
    testonly = True,
    srcs = [":gpu_test_kernels"],
    outs = ["gpu_test_kernels.fatbin"],
    cmd = """
      STATIC_LIBRARY=""
      for src in $(SRCS); do
        if [[ $$src == *.a ]]; then
          STATIC_LIBRARY=$$src
          break
        fi
      done

      if [[ -z $$STATIC_LIBRARY ]]; then
        echo "No static library found in $(SRCS)" >&2
        exit 1
      fi

      $(OBJCOPY) "--dump-section=.nv_fatbin=$@" "$$STATIC_LIBRARY" || true
      
      if [ ! -f "$@" ]; then
        # binutils' objcopy doesn't return a non-zero exit code if the
        # section was not found, so we need to check for the file's existence instead.
        $(OBJCOPY) "--dump-section=.hip_fatbin=$@" "$$STATIC_LIBRARY"
      fi
    """,
    tags = ["gpu"],
    toolchains = ["@bazel_tools//tools/cpp:current_cc_toolchain"],
)

cc_library(
    name = "gpu_test_kernels_fatbin",
    testonly = True,
    srcs = ["gpu_test_kernels_fatbin.cc"],
    hdrs = ["gpu_test_kernels_fatbin.h"],
    data = [":gpu_test_kernels_fatbin_extractor"],
    local_defines = [
        "FATBIN_SRC=\\\"$(rootpath :gpu_test_kernels_fatbin_extractor)\\\"",
    ],
    tags = ["gpu"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:path",
    ],
)

xla_test(
    name = "gpu_kernel_test",
    srcs = ["gpu_kernel_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_test_kernels",
        ":gpu_test_kernels_fatbin",
        "//xla/service:platform_util",
        "//xla/stream_executor",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:typed_kernel_factory",
        "//xla/stream_executor/rocm:rocm_platform_id",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "gpu_command_buffer_test",
    srcs = ["gpu_command_buffer_test.cc"],
    backends = ["gpu"],
    deps = [
        ":gpu_command_buffer",
        ":gpu_driver_header",
        ":gpu_test_kernels",
        ":gpu_types_header",
        "//xla/service:platform_util",
        "//xla/stream_executor",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:trace_command_buffer_factory",
        "//xla/stream_executor:typed_kernel_factory",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/rocm:rocm_platform_id",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
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

xla_test(
    name = "memcpy_test",
    srcs = ["memcpy_test.cc"],
    backends = ["gpu"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:platform_manager",
        "//xla/tsl/lib/core:status_test_util",
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
    name = "stream_search_test",
    size = "small",
    srcs = ["stream_search_test.cc"],
    backends = ["gpu"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        "//xla/stream_executor",
        "//xla/stream_executor:stream_finder",
        "//xla/stream_executor/host:host_platform",
        "@com_google_absl//absl/status:statusor",
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
    name = "gpu_executor_test",
    size = "small",
    srcs = ["gpu_executor_test.cc"],
    backends = ["gpu"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        "//xla/service:platform_util",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ] + if_cuda([
        "//xla/stream_executor/cuda:cuda_platform",
    ]) + if_rocm([
        "//xla/stream_executor/rocm:rocm_platform",
    ]),
)

xla_test(
    name = "gpu_device_info_test",
    srcs = ["gpu_device_info_test.cc"],
    backends = ["gpu"],
    data = [
        "//xla/tools/hlo_opt:gpu_specs/a100_pcie_80.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/a100_sxm_40.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/a100_sxm_80.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/a6000.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/h100_pcie.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/h100_sxm.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/mi200.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/p100.txtpb",
        "//xla/tools/hlo_opt:gpu_specs/v100.txtpb",
    ],
    deps = [
        "//xla/service:platform_util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)
