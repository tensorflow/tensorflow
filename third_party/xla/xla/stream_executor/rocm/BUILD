"""ROCm-platform specific StreamExecutor support code."""

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_hipblaslt",
    "rocm_library",
)
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/service/gpu:build_defs.bzl", "get_cub_sort_kernel_types")
load(
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl:tsl.bzl",
    "internal_visibility",
)
load("//xla/tsl/platform:build_config_root.bzl", "if_static")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "rocm_diagnostics",
    srcs = ["rocm_diagnostics.cc"],
    hdrs = ["rocm_diagnostics.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "rocm_context",
    srcs = ["rocm_context.cc"],
    hdrs = ["rocm_context.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_driver_wrapper",
        ":rocm_status",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:context_map",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

cc_library(
    name = "rocm_driver_wrapper",
    hdrs = ["rocm_driver_wrapper.h"],
    defines = {"__HIP_DISABLE_CPP_FUNCTIONS__": "1"},
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        "//xla/tsl/platform:env",
        "@local_config_rocm//rocm:hip",  # buildcleaner: keep
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:dso_loader",
    ],
)

cc_library(
    name = "rocm_event",
    srcs = ["rocm_event.cc"],
    hdrs = ["rocm_event.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_driver_wrapper",
        ":rocm_status",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:event",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

xla_test(
    name = "rocm_event_test",
    srcs = ["rocm_event_test.cc"],
    backends = ["gpu"],
    tags = ["rocm-only"],
    deps = [
        ":rocm_event",
        ":rocm_executor",
        ":rocm_platform_id",
        "//xla/stream_executor:event",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

cc_library(
    name = "rocm_executor",
    srcs = ["rocm_executor.cc"],
    hdrs = ["rocm_executor.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_command_buffer",
        ":rocm_context",
        ":rocm_diagnostics",
        ":rocm_driver_wrapper",
        ":rocm_event",
        ":rocm_kernel",
        ":rocm_platform_id",
        ":rocm_status",
        ":rocm_stream",
        ":rocm_timer",
        ":rocm_version_parser",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:event",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:generic_memory_allocation",
        "//xla/stream_executor:generic_memory_allocator",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_allocator",
        "//xla/stream_executor:module_spec",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:read_numa_node",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:numbers",
    ],
    alwayslink = True,
)

xla_test(
    name = "rocm_executor_test",
    srcs = ["rocm_executor_test.cc"],
    backends = ["gpu"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_executor",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_test_kernels",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "rocm_kernel",
    srcs = ["rocm_kernel.cc"],
    hdrs = ["rocm_kernel.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_driver_wrapper",
        ":rocm_status",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

xla_test(
    name = "rocm_kernel_test",
    srcs = ["rocm_kernel_test.cc"],
    backends = ["gpu"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_test_kernels",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "rocm_platform",
    srcs = ["rocm_platform.cc"],
    hdrs = ["rocm_platform.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_diagnostics",
        ":rocm_driver_wrapper",
        ":rocm_executor",
        ":rocm_platform_id",
        ":rocm_status",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_rocm//rocm:rocm_headers",
    ],
    alwayslink = True,  # Registers itself with the PlatformManager.
)

cc_library(
    name = "rocm_platform_id",
    srcs = ["rocm_platform_id.cc"],
    hdrs = ["rocm_platform_id.h"],
    deps = ["//xla/stream_executor:platform"],
)

cc_library(
    name = "rocblas_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = if_static([
        "@local_config_rocm//rocm:rocblas",
    ]),
)

cc_library(
    name = "rocblas_wrapper",
    hdrs = ["rocblas_wrapper.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_executor",
        "//xla/tsl/platform:env",
        "//xla/tsl/util:determinism_for_kernels",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:dso_loader",
    ],
    alwayslink = True,
)

cc_library(
    name = "rocblas_plugin",
    srcs = ["rocm_blas.cc"],
    hdrs = ["rocm_blas.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":hipblas_lt_header",
        ":rocblas_if_static",
        ":rocblas_wrapper",
        ":rocm_complex_converters",
        ":rocm_executor",
        ":rocm_helpers",
        ":rocm_platform_id",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:host_or_device_scalar",
        "//xla/stream_executor:numeric_options",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:determinism_hdr_lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_config_rocm//rocm:rocm_headers",
    ],
    alwayslink = True,
)

cc_library(
    name = "hipfft_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = if_static([
        "@local_config_rocm//rocm:hipfft",
    ]),
)

cc_library(
    name = "rocm_solver_context",
    srcs = ["rocm_solver_context.cc"],
    hdrs = ["rocm_solver_context.h"],
    local_defines = [
        "TENSORFLOW_USE_ROCM=1",
    ],
    tags = [
        "gpu",
        "manual",
        "rocm-only",
    ],
    deps = [
        ":hipsolver_wrapper",
        ":rocblas_wrapper",
        ":rocsolver_wrapper",
        "//xla:comparison_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:gpu_solver_context",
        "//xla/stream_executor:stream",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

cc_library(
    name = "hipfft_plugin",
    srcs = ["rocm_fft.cc"],
    hdrs = ["rocm_fft.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":hipfft_if_static",
        ":rocm_complex_converters",
        ":rocm_platform_id",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/status",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:dso_loader",
    ],
    alwayslink = True,
)

cc_library(
    name = "miopen_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = if_static([
        "@local_config_rocm//rocm:miopen",
    ]),
)

cc_library(
    name = "miopen_plugin",
    srcs = ["rocm_dnn.cc"],
    hdrs = ["rocm_dnn.h"],
    copts = [
        # STREAM_EXECUTOR_CUDNN_WRAP would fail on Clang with the default
        # setting of template depth 256
        "-ftemplate-depth-512",
    ],
    tags = [
        "gpu",
        "rocm-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":miopen_if_static",  # build_cleaner: keep
        ":rocm_diagnostics",
        ":rocm_executor",
        ":rocm_helpers",
        ":rocm_platform_id",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:numeric_options",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:determinism_for_kernels",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:dso_loader",
        "@local_tsl//tsl/platform:hash",
    ],
    alwayslink = True,
)

cc_library(
    name = "hiprand_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = if_static([
        "@local_config_rocm//rocm:hiprand",
    ]),
)

cc_library(
    name = "hipsparse_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = if_static([
        "@local_config_rocm//rocm:hipsparse",
    ]),
)

cc_library(
    name = "hipsparse_wrapper",
    srcs = ["hipsparse_wrapper.h"],
    hdrs = ["hipsparse_wrapper.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":hipsparse_if_static",
        ":rocm_executor",
        ":rocm_platform_id",
        "//xla/tsl/platform:env",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:dso_loader",
    ],
    alwayslink = True,
)

cc_library(
    name = "rocsolver_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = if_static([
        "@local_config_rocm//rocm:rocsolver",
    ]),
)

cc_library(
    name = "rocsolver_wrapper",
    srcs = ["rocsolver_wrapper.h"],
    hdrs = ["rocsolver_wrapper.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_executor",
        ":rocm_platform_id",
        ":rocsolver_if_static",
        "//xla/tsl/platform:env",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:dso_loader",
    ],
    alwayslink = True,
)

cc_library(
    name = "hipsolver_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = if_static([
        "@local_config_rocm//rocm:hipsolver",
    ]),
)

cc_library(
    name = "hipsolver_wrapper",
    hdrs = ["hipsolver_wrapper.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":hipsolver_if_static",
        ":rocm_executor",
        ":rocm_platform_id",
        "//xla/tsl/platform:env",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:dso_loader",
    ],
    alwayslink = True,
)

cc_library(
    name = "hipblaslt_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = if_rocm_hipblaslt([
        "@local_config_rocm//rocm:hipblaslt",
    ]),
)

cc_library(
    name = "amdhipblaslt_plugin",
    srcs = ["hip_blas_lt.cc"],
    hdrs = [
        "hip_blas_lt.h",
        "hip_blas_utils.h",
        "hipblaslt_wrapper.h",
    ],
    defines = {"__HIP_DISABLE_CPP_FUNCTIONS__": "1"},
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":hip_blas_utils",
        ":hipblas_lt_header",
        ":rocblas_plugin",
        ":rocm_executor",
        ":rocm_platform_id",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:host_or_device_scalar",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@eigen_archive//:eigen3",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:dso_loader",
        "@local_tsl//tsl/platform:ml_dtypes",
    ] + if_static([
        ":hipblaslt_if_static",
    ]),
    alwayslink = True,
)

cc_library(
    name = "hipblas_lt_header",
    hdrs = [
        "hip_blas_lt.h",
        "hip_blas_utils.h",
        "hipblaslt_wrapper.h",
    ],
    tags = [
        "gpu",
        "rocm-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:types",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:host_or_device_scalar",
        "//xla/stream_executor:stream",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/status",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:dso_loader",
    ],
)

cc_library(
    name = "hip_blas_utils",
    srcs = ["hip_blas_utils.cc"],
    hdrs = ["hip_blas_utils.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":hipblas_lt_header",
        ":rocblas_plugin",
        "//xla/stream_executor:blas",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "roctracer_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = if_static([
        "@local_config_rocm//rocm:roctracer",
    ]),
)

cc_library(
    name = "roctracer_wrapper",
    srcs = ["roctracer_wrapper.h"],
    hdrs = ["roctracer_wrapper.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_executor",
        "//xla/tsl/platform:env",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:dso_loader",
    ],
    alwayslink = True,
)

rocm_library(
    name = "rocm_helpers",
    srcs = ["rocm_helpers.cu.cc"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
    ],
    alwayslink = True,
)

cc_library(
    name = "rocm_complex_converters",
    hdrs = ["rocm_complex_converters.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        "@com_google_absl//absl/log:check",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

cc_library(
    name = "all_runtime",
    tags = [
        "gpu",
        "rocm-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":all_reduce_kernel_rocm",
        ":amdhipblaslt_plugin",
        ":buffer_comparator_kernel_rocm",
        ":hipfft_plugin",
        ":make_batch_pointers_kernel_rocm",
        ":miopen_plugin",
        ":ragged_all_to_all_kernel_rocm",
        ":rocblas_plugin",
        ":rocm_helpers",
        ":rocm_platform",
        ":topk_kernel_rocm",
    ] + [":cub_sort_kernel_rocm_" + suffix for suffix in get_cub_sort_kernel_types()],
    alwayslink = 1,
)

cc_library(
    name = "stream_executor_rocm",
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_platform_id",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/host:host_platform_id",
        "@local_config_rocm//rocm:rocm_rpath",
    ] + if_static(
        [":all_runtime"],
    ),
)

cc_library(
    name = "rocm_version_parser",
    srcs = ["rocm_version_parser.cc"],
    hdrs = ["rocm_version_parser.h"],
    deps = [
        "//xla/stream_executor:semantic_version",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_cc_test(
    name = "rocm_version_parser_test",
    srcs = ["rocm_version_parser_test.cc"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_version_parser",
        "//xla/stream_executor:semantic_version",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

cc_library(
    name = "rocm_stream",
    srcs = ["rocm_stream.cc"],
    hdrs = ["rocm_stream.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_driver_wrapper",
        ":rocm_event",
        ":rocm_kernel",
        ":rocm_status",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:event",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_common",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

xla_test(
    name = "rocm_stream_test",
    srcs = ["rocm_stream_test.cc"],
    backends = ["gpu"],
    tags = ["rocm-only"],
    deps = [
        ":rocm_event",
        ":rocm_executor",
        ":rocm_platform_id",
        ":rocm_stream",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:typed_kernel_factory",
        "//xla/stream_executor/gpu:gpu_test_kernels",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "rocm_timer",
    srcs = ["rocm_timer.cc"],
    hdrs = ["rocm_timer.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_driver_wrapper",
        ":rocm_event",
        ":rocm_status",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

xla_test(
    name = "rocm_timer_test",
    srcs = ["rocm_timer_test.cc"],
    backends = ["gpu"],
    tags = ["rocm-only"],
    deps = [
        ":rocm_executor",
        ":rocm_platform_id",
        ":rocm_timer",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:typed_kernel_factory",
        "//xla/stream_executor/gpu:gpu_test_kernels",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "rocm_status",
    srcs = ["rocm_status.cc"],
    hdrs = ["rocm_status.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

xla_cc_test(
    name = "rocm_status_test",
    srcs = ["rocm_status_test.cc"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_status",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

cc_library(
    name = "rocm_command_buffer",
    srcs = [
        "rocm_command_buffer.cc",
    ],
    hdrs = ["rocm_command_buffer.h"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_driver_wrapper",
        ":rocm_kernel",
        ":rocm_status",
        "//xla/stream_executor:bit_pattern",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_command_buffer",
        "//xla/stream_executor/gpu:scoped_gpu_graph_exec",
        "//xla/stream_executor/gpu:scoped_update_mode",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:casts",
    ],
)

rocm_library(
    name = "buffer_comparator_kernel_rocm",
    srcs = [
        "buffer_comparator_kernel_rocm.cu.cc",
        "//xla/stream_executor/gpu:buffer_comparator_kernel_lib.cu.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_platform_id",
        "//xla:shape_util",
        "//xla:types",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor/gpu:buffer_comparator_kernel",
        "//xla/stream_executor/gpu:gpu_kernel_registry",
        "//xla/stream_executor/platform:initialize",
        "@local_config_rocm//rocm:rocm_headers",
    ],
    alwayslink = 1,
)

rocm_library(
    name = "make_batch_pointers_kernel_rocm",
    srcs = ["make_batch_pointers_kernel_rocm.cu.cc"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_platform_id",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor/gpu:gpu_kernel_registry",
        "//xla/stream_executor/gpu:make_batch_pointers_kernel",
        "//xla/stream_executor/platform:initialize",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_config_rocm//rocm:rocm_headers",
    ],
    alwayslink = 1,
)

rocm_library(
    name = "ragged_all_to_all_kernel_rocm",
    srcs = [
        "ragged_all_to_all_kernel_rocm.cc",
        "//xla/stream_executor/gpu:ragged_all_to_all_kernel_lib.cu.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_platform_id",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor/gpu:gpu_kernel_registry",
        "//xla/stream_executor/gpu:ragged_all_to_all_kernel",
        "@local_config_rocm//rocm:rocm_headers",
    ],
    alwayslink = 1,
)

rocm_library(
    name = "all_reduce_kernel_rocm",
    srcs = [
        "all_reduce_kernel_rocm.cc",
        "//xla/stream_executor/gpu:all_reduce_kernel_lib.cu.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_platform_id",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor/gpu:all_reduce_kernel",
        "//xla/stream_executor/gpu:gpu_kernel_registry",
        "@local_config_rocm//rocm:rocm_headers",
    ],
    alwayslink = 1,
)

[rocm_library(
    name = "cub_sort_kernel_rocm_{}".format(typename),
    srcs = ["cub_sort_kernel_rocm.cu.cc"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    local_defines = ["CUB_TYPE_" + typename.upper()],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/stream_executor/rocm:rocm_status",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@eigen_archive//:eigen3",
        "@local_config_rocm//rocm:rocprim",
        "@local_tsl//tsl/platform:bfloat16",
    ],
    alwayslink = 1,
) for typename in get_cub_sort_kernel_types()]

rocm_library(
    name = "topk_kernel_rocm",
    srcs = [
        "topk_kernel_rocm_bfloat16.cu.cc",
        "topk_kernel_rocm_common.cu.h",
        "topk_kernel_rocm_float.cu.cc",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        ":rocm_platform_id",
        "//xla:types",
        "//xla/stream_executor/gpu:gpu_kernel_registry",
        "//xla/stream_executor/gpu:topk_kernel",
        "//xla/tsl/lib/math:math_util",
        "@local_config_rocm//rocm:rocm_headers",
    ],
    alwayslink = 1,
)
