load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
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
    "if_cuda_newer_than",
)
load(
    "//xla:xla.bzl",
    "xla_cc_test",
)
load(
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
    "tf_additional_cuda_platform_deps",
    "tf_additional_cudnn_plugin_copts",
    "tf_additional_gpu_compilation_copts",
)
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl:tsl.bzl",
    "if_cuda_tools",
    "if_google",
    "if_nccl",
    "internal_visibility",
    "tsl_copts",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

bool_flag(
    name = "enable_libnvptxcompiler_support",
    build_setting_default = if_google(
        True,
        oss_value = False,
    ),
)

config_setting(
    name = "libnvptxcompiler_support_enabled",
    flag_values = {
        ":enable_libnvptxcompiler_support": "True",
    },
)

bool_flag(
    name = "enable_libnvjitlink_support",
    build_setting_default = if_google(
        True,
        oss_value = False,
    ),
)

config_setting(
    name = "libnvjitlink_support_enabled",
    flag_values = {
        ":enable_libnvjitlink_support": "True",
    },
)

cc_library(
    name = "cuda_platform_id",
    srcs = ["cuda_platform_id.cc"],
    hdrs = ["cuda_platform_id.h"],
    deps = ["//xla/stream_executor:platform"],
)

cc_library(
    name = "cuda_platform",
    srcs = ["cuda_platform.cc"],
    hdrs = ["cuda_platform.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps =
        [
            ":cuda_collectives",
            ":cuda_driver",
            ":cuda_executor",
            ":cuda_platform_id",
            ":cuda_runtime",
            "//xla/stream_executor",
            "//xla/stream_executor:executor_cache",
            "//xla/stream_executor:platform_manager",
            "//xla/stream_executor:stream_executor_h",
            "//xla/stream_executor/gpu:gpu_driver_header",
            "//xla/stream_executor/gpu:gpu_executor_header",
            "//xla/stream_executor/platform:initialize",
            "@com_google_absl//absl/base",
            "@com_google_absl//absl/base:core_headers",
            "@com_google_absl//absl/log",
            "@com_google_absl//absl/log:check",
            "@com_google_absl//absl/memory",
            "@com_google_absl//absl/status",
            "@com_google_absl//absl/status:statusor",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
            "@com_google_absl//absl/synchronization",
            "@local_tsl//tsl/platform:errors",
            "@local_tsl//tsl/platform:status",
            "@local_tsl//tsl/platform:statusor",
        ] + tf_additional_cuda_platform_deps(),
    alwayslink = True,  # Registers itself with the PlatformManager.
)

cc_library(
    name = "cuda_diagnostics",
    srcs = ["cuda_diagnostics.cc"],
    hdrs = ["cuda_diagnostics.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "//xla/stream_executor/gpu:gpu_diagnostics_header",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

# Buildozer can not remove dependencies inside select guards, so we have to use
# an intermediate target.
cc_library(
    name = "ptxas_wrapper",
    data = if_cuda_tools(
        ["@cuda_nvcc//:ptxas"],
        [],
    ),
)

cc_library(
    name = "nvlink_wrapper",
    data = if_cuda_tools(
        ["@cuda_nvcc//:nvlink"],
        [],
    ),
)

cc_library(
    name = "cuda_context",
    srcs = ["cuda_context.cc"],
    hdrs = ["cuda_context.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":cuda_status",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:context_map",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "cuda_driver",
    srcs = ["cuda_driver.cc"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":cuda_context",
        ":cuda_diagnostics",  # buildcleaner: keep
        ":cuda_status",
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:context_map",
        "//xla/stream_executor/gpu:gpu_diagnostics_header",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/tsl/cuda",
        "//xla/tsl/cuda:cudart",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:stacktrace",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "cuda_status",
    srcs = ["cuda_status.cc"],
    hdrs = ["cuda_status.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "cuda_runtime",
    srcs = ["cuda_runtime.cc"],
    hdrs = ["cuda_runtime.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "cuda_collectives",
    hdrs = ["cuda_collectives.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = if_nccl(
        [":cuda_collectives_impl"],
        [":cuda_collectives_stub"],
    ) + [
        "//xla/stream_executor/gpu:context",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "cuda_collectives_impl",
    srcs = [
        "cuda_collectives.cc",
        "cuda_collectives.h",
    ],
    tags = [
        "cuda-only",
        "gpu",
        "manual",
    ],
    deps = [
        ":cuda_driver",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_nccl//:nccl",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "cuda_collectives_stub",
    srcs = [
        "cuda_collectives.h",
        "cuda_collectives_stub.cc",
    ],
    deps = [
        "//xla/stream_executor/gpu:context",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "cuda_collectives_test",
    srcs = ["cuda_collectives_test.cc"],
    backends = ["gpu_any"],
    tags = ["cuda-only"],
    deps = [
        ":cuda_collectives",
        "//xla/service/gpu/runtime:nccl_api",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_test(
    name = "cuda_driver_test",
    srcs = ["cuda_driver_test.cc"],
    backends = ["gpu"],
    tags = [
        "cuda-only",
    ],
    deps = [
        ":cuda_context",
        ":cuda_diagnostics",
        ":cuda_status",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "cublas_lt_header",
    hdrs = [
        "cuda_blas_lt.h",
        "cuda_blas_utils.h",
    ],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:types",
        "//xla/stream_executor",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "cublas_plugin",
    srcs = [
        "cuda_blas.cc",
        "cuda_blas_lt.cc",
    ],
    hdrs = [
        "cuda_blas.h",
        "cuda_blas_lt.h",
    ],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_blas_utils",
        ":cuda_executor",
        ":cuda_helpers",
        ":cuda_platform_id",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:host_or_device_scalar",
        "//xla/stream_executor:numeric_options",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/cuda:cublas",
        "//xla/tsl/cuda:cublas_lt",
        "//xla/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@eigen_archive//:eigen3",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:tensor_float_32_hdr_lib",
    ] + if_static([
        "@local_tsl//tsl/platform:tensor_float_32_utils",
    ]),
    alwayslink = True,
)

cc_library(
    name = "cuda_blas_utils",
    srcs = ["cuda_blas_utils.cc"],
    hdrs = ["cuda_blas_utils.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "//xla/stream_executor",
        "//xla/stream_executor:blas",
        "//xla/tsl/cuda:cublas",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "cufft_plugin",
    srcs = ["cuda_fft.cc"],
    hdrs = ["cuda_fft.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_helpers",
        ":cuda_platform_id",
        "//xla/stream_executor",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/cuda:cufft",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,
)

cuda_library(
    name = "delay_kernel_cuda",
    srcs = [
        "delay_kernel.h",
        "delay_kernel_cuda.cu.cc",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = internal_visibility([
        "//xla/stream_executor:__subpackages__",
    ]),
    deps = [
        "//xla/stream_executor",
        "//xla/stream_executor:typed_kernel_factory",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_semaphore",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "cudnn_plugin",
    srcs = ["cuda_dnn.cc"],
    hdrs = ["cuda_dnn.h"],
    copts = tf_additional_cudnn_plugin_copts(),
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_diagnostics",
        ":cuda_driver",
        ":cuda_executor",
        ":cuda_platform_id",
        ":cudnn_frontend_helpers",
        "//xla/stream_executor",
        "//xla/stream_executor:data_type",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:numeric_options",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_diagnostics_header",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/cuda:cudnn",
        "//xla/tsl/protobuf:dnn_proto_cc",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@cudnn_frontend_archive//:cudnn_frontend",
        "@eigen_archive//:eigen3",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudnn_header",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:tensor_float_32_hdr_lib",
        "@local_tsl//tsl/platform:tensor_float_32_utils",
    ],
    alwayslink = True,
)

cc_library(
    name = "cuda_kernel",
    srcs = ["cuda_kernel.cc"],
    hdrs = ["cuda_kernel.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_kernel_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_test(
    name = "cuda_kernel_test",
    srcs = ["cuda_kernel_test.cc"],
    backends = ["gpu_any"],
    tags = ["cuda-only"],
    deps = [
        ":cuda_kernel",
        ":cuda_runtime",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_test_kernels_cuda",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cuda_library(
    name = "command_buffer_kernels",
    srcs = [
        "command_buffer_kernels.cc",
        "command_buffer_kernels.cu.cc",
    ],
    tags = ["cuda-only"],
    deps = [
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor/gpu:gpu_types_header",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "cuda_helpers",
    hdrs = ["cuda_helpers.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "@com_google_absl//absl/log:check",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "cuda_event",
    srcs = ["cuda_event.cc"],
    hdrs = ["cuda_event.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":cuda_status",
        "//xla/stream_executor:event",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "cuda_event_test",
    srcs = ["cuda_event_test.cc"],
    backends = ["gpu"],
    tags = ["cuda-only"],
    deps = [
        ":cuda_event",
        ":cuda_executor",
        ":cuda_platform_id",
        "//xla/stream_executor:event",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

# This target serves to expose a single variable to all other kinds of
# targets and should stay minimal (no dependencies).
cc_library(
    name = "ptx_compiler_support",
    srcs = ["ptx_compiler_support.cc"],
    hdrs = ["ptx_compiler_support.h"],
    local_defines = select({
        ":libnvptxcompiler_support_enabled": [
            "LIBNVPTXCOMPILER_SUPPORT=true",
        ],
        "//conditions:default": [
            "LIBNVPTXCOMPILER_SUPPORT=false",
        ],
    }),
)

cc_library(
    name = "ptx_compiler_helpers",
    srcs = ["ptx_compiler_helpers.cc"],
    hdrs = ["ptx_compiler_helpers.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "ptx_compiler_stub",
    srcs = [
        "ptx_compiler.h",
        "ptx_compiler_stub.cc",
    ],
    deps = [
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "ptx_compiler_impl",
    srcs = [
        "ptx_compiler.h",
        "ptx_compiler_impl.cc",
    ],
    tags = ["manual"],
    deps = [
        ":ptx_compiler_helpers",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:nvptxcompiler",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "ptx_compiler",
    hdrs = ["ptx_compiler.h"],
    tags = ["cuda-only"],
    deps = select({
        ":libnvptxcompiler_support_enabled": [":ptx_compiler_impl"],
        "//conditions:default": [":ptx_compiler_stub"],
    }) + [
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_test(
    name = "cuda_platform_test",
    srcs = ["cuda_platform_test.cc"],
    backends = ["gpu"],
    tags = ["cuda-only"],
    deps = [
        ":cuda_platform",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "ptx_compiler_test",
    srcs = ["ptx_compiler_test.cc"],
    tags = [
        "cuda-only",
        # TODO(b/343996893): Figure out whether msan reports a false positive or not.
        "nomsan",
    ],
    deps = [
        ":ptx_compiler",
        ":ptx_compiler_support",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "nvjitlink_support",
    srcs = ["nvjitlink_support.cc"],
    hdrs = ["nvjitlink_support.h"],
    local_defines = select({
        ":libnvjitlink_support_enabled": [
            "LIBNVJITLINK_SUPPORT=true",
        ],
        "//conditions:default": [
            "LIBNVJITLINK_SUPPORT=false",
        ],
    }) + if_cuda_newer_than(
        "12_0",
        ["CUDA_SUPPORTS_NVJITLINK=true"],
        ["CUDA_SUPPORTS_NVJITLINK=false"],
    ),
)

cc_library(
    name = "nvjitlink_stub",
    srcs = [
        "nvjitlink.h",
        "nvjitlink_stub.cc",
    ],
    deps = [
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "nvjitlink_impl",
    srcs = [
        "nvjitlink.h",
        "nvjitlink_impl.cc",
    ],
    tags = ["manual"],
    deps = [
        ":ptx_compiler_helpers",
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:nvjitlink",  # buildcleaner: keep
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

# Since select() can't be nested, we need to wrap the cuda_newer_than check in a separate
# library target.
cc_library(
    name = "nvjitlink_cuda_supported",
    # Even though the macro is called `*_newer_than`, it does a greater-than-or-equal-to comparison.
    deps = if_cuda_newer_than(
        "12_0",
        [":nvjitlink_impl"],
        [":nvjitlink_stub"],
    ),
)

cc_library(
    name = "nvjitlink",
    hdrs = [
        "nvjitlink.h",
    ],
    deps = select({
        ":libnvjitlink_support_enabled": [":nvjitlink_cuda_supported"],
        "//conditions:default": [":nvjitlink_stub"],
    }) + [
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "nvjitlink_test",
    srcs = ["nvjitlink_test.cc"],
    args = if_google([
        # nvjitlink allocates memory and only keeps a pointer past the usual offset of 1024 bytes;
        # so we need to increase the max pointer offset. -1 means no limit.
        # This is only relevant for Google's HeapLeakChecker. The newer Leak sanitizer doesn't
        # have this issue.
        "--heap_check_max_pointer_offset=-1",
    ]),
    # The test fails with msan/tsan since its introduction.
    tags = [
        "nomsan",
        "notsan",
    ],
    deps = [
        ":nvjitlink",
        ":nvjitlink_support",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "cuda_asm_compiler",
    srcs = ["cuda_asm_compiler.cc"],
    hdrs = ["cuda_asm_compiler.h"],
    copts = tf_additional_gpu_compilation_copts(),
    # copybara:uncomment_begin
    # data = [
    # "@local_config_cuda//cuda:runtime_fatbinary",
    # "@local_config_cuda//cuda:runtime_nvlink",
    # "@local_config_cuda//cuda:runtime_ptxas",
    # ],
    # copybara:uncomment_end
    # copybara:comment_begin
    data = if_cuda_tools([
        "@cuda_nvcc//:fatbinary",
        "@cuda_nvcc//:nvlink",
        "@cuda_nvcc//:ptxas",
    ]),
    # copybara:comment_end
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = internal_visibility([
        "//third_party/py/jax:__subpackages__",
        "//tensorflow/compiler/mlir/tools/kernel_gen:__subpackages__",
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
    ]),
    deps = [
        ":cuda_driver",  # buildcleaner: keep
        ":cuda_status",
        ":ptx_compiler",
        ":ptx_compiler_helpers",
        ":ptx_compiler_support",
        "//xla:status_macros",
        "//xla:util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:gpu_asm_opts",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:cuda_root_path",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:regexp",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:subprocess",
    ],
)

cc_library(
    name = "cuda_executor",
    srcs = [
        "cuda_executor.cc",
        "delay_kernel.h",
    ],
    hdrs = [
        "cuda_executor.h",
    ],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":cuda_collectives",
        ":cuda_context",
        ":cuda_event",  # buildcleaner: keep
        ":cuda_kernel",  # buildcleaner: keep
        ":cuda_platform_id",
        ":cuda_runtime",
        ":cuda_status",
        ":cuda_stream",
        ":cuda_timer",
        ":cuda_version_parser",
        ":delay_kernel_cuda",
        "//xla/stream_executor",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:event",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:host_memory_allocation",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:module_spec",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:gpu_command_buffer",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_kernel_header",
        "//xla/stream_executor/gpu:gpu_semaphore",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/stream_executor/gpu:read_numa_node",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,
)

xla_test(
    name = "cuda_executor_test",
    srcs = ["cuda_executor_test.cc"],
    backends = ["gpu"],
    tags = ["cuda-only"],
    deps = [
        ":cuda_executor",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "all_runtime",
    copts = tsl_copts(),
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cublas_plugin",
        ":cuda_driver",
        ":cuda_platform",
        ":cudnn_plugin",
        ":cufft_plugin",
        "//xla/tsl/cuda:cusolver",
        "//xla/tsl/cuda:cusparse",
        "//xla/tsl/cuda:tensorrt_rpath",
    ],
    alwayslink = 1,
)

# OSX framework for device driver access
cc_library(
    name = "IOKit",
    linkopts = ["-framework IOKit"],
)

cc_library(
    name = "stream_executor_cuda",
    tags = ["cuda-only"],
    deps = [
        ":cuda_platform_id",
        "//xla/stream_executor",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor/host:host_platform_id",
        "//xla/stream_executor/rocm:rocm_platform_id",
    ] + if_google(
        select({
            # copybara:uncomment_begin(different config setting in OSS)
            # "//tools/cc_target_os:gce": [],
            # copybara:uncomment_end_and_comment_begin
            "//conditions:default": [
                "@local_config_cuda//cuda:cuda_runtime",  # buildcleaner: keep
                ":cuda_platform",
            ],
        }),
        [
            "//xla/tsl/cuda:cudart",
        ] + select({
            "//xla/tsl:macos": ["IOKit"],
            "//conditions:default": [],
        }),
    ),
)

cc_library(
    name = "cudnn_frontend_helpers",
    srcs = ["cudnn_frontend_helpers.cc"],
    hdrs = ["cudnn_frontend_helpers.h"],
)

cc_library(
    name = "ptx_compilation_method",
    hdrs = ["ptx_compilation_method.h"],
    deps = ["@com_google_absl//absl/strings"],
)

cc_library(
    name = "ptx_linking_method",
    hdrs = ["ptx_linking_method.h"],
    deps = ["@com_google_absl//absl/strings"],
)

cc_library(
    name = "cuda_version_parser",
    srcs = ["cuda_version_parser.cc"],
    hdrs = ["cuda_version_parser.h"],
    deps = [
        "//xla/stream_executor:semantic_version",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "cuda_version_parser_test",
    srcs = ["cuda_version_parser_test.cc"],
    deps = [
        ":cuda_version_parser",
        "//xla/stream_executor:semantic_version",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "cuda_stream",
    srcs = ["cuda_stream.cc"],
    hdrs = ["cuda_stream.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":cuda_event",
        ":cuda_status",
        "//xla/stream_executor:event",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "cuda_timer",
    srcs = [
        "cuda_timer.cc",
    ],
    hdrs = ["cuda_timer.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":cuda_event",
        ":cuda_status",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:stream",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:gpu_semaphore",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)
