# Description:
#   ROCm-platform specific StreamExecutor support code.

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_hipblaslt",
    "rocm_library",
)
load("@local_tsl//tsl/platform:build_config_root.bzl", "if_static")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl:tsl.bzl",
    "if_google",
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

cc_library(
    name = "rocm_diagnostics",
    srcs = ["rocm_diagnostics.cc"],
    hdrs = ["rocm_diagnostics.h"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        "//xla/stream_executor/gpu:gpu_diagnostics_header",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
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
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":rocm_driver_wrapper",
        ":rocm_status",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:context_map",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "rocm_driver_wrapper",
    hdrs = ["rocm_driver_wrapper.h"],
    defines = {"__HIP_DISABLE_CPP_FUNCTIONS__": "1"},
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(klucke): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        "@local_config_rocm//rocm:hip",  # buildcleaner: keep
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:dso_loader",
        "@local_tsl//tsl/platform:env",
    ],
)

cc_library(
    name = "rocm_driver",
    srcs = ["rocm_driver.cc"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":rocm_context",
        ":rocm_driver_wrapper",
        ":rocm_status",
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:context_map",
        "//xla/stream_executor/gpu:gpu_diagnostics_header",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_config_rocm//rocm:hip",  # buildcleaner: keep
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:stacktrace",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "rocm_runtime",
    srcs = ["rocm_runtime.cc"],
    hdrs = ["rocm_runtime.h"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":rocm_driver_wrapper",
        ":rocm_status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "rocm_event",
    srcs = ["rocm_event.cc"],
    hdrs = ["rocm_event.h"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":rocm_driver_wrapper",
        ":rocm_status",
        "//xla/stream_executor:event",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:gpu_event_header",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "rocm_executor",
    srcs = ["rocm_executor.cc"],
    hdrs = ["rocm_executor.h"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":rocm_context",
        ":rocm_driver",
        ":rocm_driver_wrapper",
        ":rocm_event",
        ":rocm_kernel",
        ":rocm_platform_id",
        ":rocm_runtime",
        ":rocm_status",
        ":rocm_stream",
        ":rocm_timer",
        ":rocm_version_parser",
        "//xla/stream_executor",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:event",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:host_memory_allocation",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:module_spec",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:gpu_command_buffer",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_event",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_kernel_header",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/stream_executor/gpu:read_numa_node",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/stream_executor/platform:initialize",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,
)

xla_test(
    name = "rocm_executor_test",
    srcs = ["rocm_executor_test.cc"],
    backends = ["gpu_amd_any"],
    tags = ["rocm-only"],
    deps = [
        ":rocm_executor",
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
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_kernel_header",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_test(
    name = "rocm_kernel_test",
    srcs = ["rocm_kernel_test.cc"],
    backends = ["gpu_amd_any"],
    deps = [
        ":rocm_kernel",
        ":rocm_runtime",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_test_kernels",
        "@com_google_googletest//:gtest_main",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "command_buffer_kernels",
    srcs = ["command_buffer_kernels.cc"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        "//xla/stream_executor:kernel_spec",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "rocm_platform",
    srcs = ["rocm_platform.cc"],
    hdrs = ["rocm_platform.h"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_driver",
        ":rocm_executor",
        ":rocm_platform_id",
        ":rocm_runtime",
        "//xla/stream_executor",  # buildcleaner: keep
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/platform:initialize",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:errors",
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
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
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
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":rocblas_if_static",
        ":rocm_executor",
        ":rocm_platform_id",
        "//xla/tsl/util:determinism_for_kernels",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:dso_loader",
        "@local_tsl//tsl/platform:env",
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
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":hipblas_lt_header",
        ":rocblas_if_static",
        ":rocblas_wrapper",
        ":rocm_complex_converters",
        ":rocm_executor",
        ":rocm_helpers",
        ":rocm_platform_id",
        "//xla/stream_executor",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:host_or_device_scalar",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/util:determinism_hdr_lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:logging",
    ],
    alwayslink = True,
)

cc_library(
    name = "hipfft_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = if_static([
        "@local_config_rocm//rocm:hipfft",
    ]),
)

cc_library(
    name = "hipfft_plugin",
    srcs = ["rocm_fft.cc"],
    hdrs = ["rocm_fft.h"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":hipfft_if_static",
        ":rocm_complex_converters",
        ":rocm_platform_id",
        "//xla/stream_executor",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/stream_executor/gpu:gpu_kernel_header",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/stream_executor/platform:initialize",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:dso_loader",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
    ],
    alwayslink = True,
)

cc_library(
    name = "miopen_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
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
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":miopen_if_static",
        ":rocm_diagnostics",
        ":rocm_driver",
        ":rocm_executor",
        ":rocm_helpers",
        ":rocm_platform_id",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/util:determinism_for_kernels",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:dso_loader",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:env_impl",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:hash",
        "@local_tsl//tsl/platform:logging",
    ],
    alwayslink = True,
)

cc_library(
    name = "hiprand_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = if_static([
        "@local_config_rocm//rocm:hiprand",
    ]),
)

cc_library(
    name = "hipsparse_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
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
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":hipsparse_if_static",
        ":rocm_executor",
        ":rocm_platform_id",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:dso_loader",
        "@local_tsl//tsl/platform:env",
    ],
    alwayslink = True,
)

cc_library(
    name = "rocsolver_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
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
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":rocm_executor",
        ":rocm_platform_id",
        ":rocsolver_if_static",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:dso_loader",
        "@local_tsl//tsl/platform:env",
    ],
    alwayslink = True,
)

cc_library(
    name = "hipsolver_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
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
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":hipsolver_if_static",
        ":rocm_executor",
        ":rocm_platform_id",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:dso_loader",
        "@local_tsl//tsl/platform:env",
    ],
    alwayslink = True,
)

cc_library(
    name = "hipblaslt_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
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
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
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
        "//xla/stream_executor",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:host_or_device_scalar",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "@com_google_absl//absl/status",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:dso_loader",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
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
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//xla:types",
        "//xla/stream_executor",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:host_or_device_scalar",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "@com_google_absl//absl/status",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:dso_loader",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "hip_blas_utils",
    srcs = ["hip_blas_utils.cc"],
    hdrs = ["hip_blas_utils.h"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":hipblas_lt_header",
        ":rocblas_plugin",
        "//xla/stream_executor:blas",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "roctracer_if_static",
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
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
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":rocm_executor",
        ":rocm_platform_id",
        ":roctracer_if_static",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:dso_loader",
        "@local_tsl//tsl/platform:env",
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
    copts = tsl_copts(),
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":amdhipblaslt_plugin",
        ":hipfft_plugin",
        ":miopen_plugin",
        ":rocblas_plugin",
        ":rocm_driver",
        ":rocm_helpers",
        ":rocm_platform",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rocm_rpath",
    linkopts = select({
        "//conditions:default": [
            "-Wl,-rpath,../local_config_rocm/rocm/rocm/lib",
        ],
    }),
)

cc_library(
    name = "stream_executor_rocm",
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":rocm_platform_id",
        ":rocm_rpath",
        "//xla/stream_executor",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/host:host_platform_id",
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

cc_test(
    name = "rocm_version_parser_test",
    srcs = ["rocm_version_parser_test.cc"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":rocm_version_parser",
        "//xla/stream_executor:semantic_version",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "rocm_stream",
    srcs = ["rocm_stream.cc"],
    hdrs = ["rocm_stream.h"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":rocm_driver_wrapper",
        ":rocm_status",
        "//xla/stream_executor:event",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:gpu_event",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "rocm_timer",
    srcs = ["rocm_timer.cc"],
    hdrs = ["rocm_timer.h"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":rocm_driver_wrapper",
        ":rocm_status",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:gpu_event",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "rocm_status",
    srcs = ["rocm_status.cc"],
    hdrs = ["rocm_status.h"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

cc_test(
    name = "rocm_status_test",
    srcs = ["rocm_status_test.cc"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    deps = [
        ":rocm_status",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
    ],
)
