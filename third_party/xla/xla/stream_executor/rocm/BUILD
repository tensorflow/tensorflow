# Description:
#   ROCm-platform specific StreamExecutor support code.
# buildifier: disable=out-of-order-load

# buildifier: disable=out-of-order-load

load(
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)

# copybara:comment_begin(oss-only)
load("//xla/stream_executor/rocm:build_defs.bzl", "rocm_embedded_test_modules")

# copybara:comment_end
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_hipblaslt",
    "if_rocm_is_configured",
    "rocm_library",
)
load("//xla/tsl:tsl.bzl", "internal_visibility", "tsl_copts")
load("@local_tsl//tsl/platform:build_config_root.bzl", "if_static")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

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
    srcs = if_rocm_is_configured(["rocm_diagnostics.cc"]),
    hdrs = if_rocm_is_configured(["rocm_diagnostics.h"]),
    deps = if_rocm_is_configured([
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xla/stream_executor/gpu:gpu_diagnostics_header",
        "//xla/stream_executor/platform",
        "@local_tsl//tsl/platform:platform_port",
    ]),
)

cc_library(
    name = "rocm_driver",
    srcs = if_rocm_is_configured(["rocm_driver.cc"]),
    hdrs = if_rocm_is_configured([
        "rocm_driver_wrapper.h",
        "rocm_driver.h",
    ]),
    deps = if_rocm_is_configured([
        ":rocm_diagnostics",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:stacktrace",
    ]),
)

cc_library(
    name = "rocm_runtime",
    srcs = if_rocm_is_configured(["rocm_runtime.cc"]),
    hdrs = if_rocm_is_configured([
        "rocm_driver_wrapper.h",
        "rocm_driver.h",
    ]),
    deps = if_rocm_is_configured([
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_runtime_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/stream_executor/gpu:gpu_driver_header",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:statusor",
    ]),
)

cc_library(
    name = "rocm_collectives",
    srcs = if_rocm_is_configured(["rocm_collectives.cc"]),
    deps = if_rocm_is_configured([
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//xla/stream_executor/gpu:gpu_collectives_header",
        "//xla/stream_executor/gpu:gpu_driver_header",
    ]),
)

cc_library(
    name = "rocm_activation",
    srcs = [],
    hdrs = if_rocm_is_configured(["rocm_activation.h"]),
    deps = if_rocm_is_configured([
        ":rocm_driver",
        "@local_config_rocm//rocm:rocm_headers",
        "//xla/stream_executor",
        "//xla/stream_executor:stream_executor_interface",
        "//xla/stream_executor/gpu:gpu_activation",
        "//xla/stream_executor/platform",
    ]),
)

cc_library(
    name = "rocm_event",
    srcs = if_rocm_is_configured(["rocm_event.cc"]),
    deps = if_rocm_is_configured([
        ":rocm_driver",
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_event_header",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_stream_header",
    ]),
)

cc_library(
    name = "rocm_executor",
    srcs = if_rocm_is_configured(["rocm_executor.cc"]),
    deps = if_rocm_is_configured([
        ":rocm_diagnostics",
        ":rocm_driver",
        ":rocm_event",
        ":rocm_kernel",
        ":rocm_platform_id",
        ":rocm_runtime",
        ":rocm_collectives",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings",
        "//xla/stream_executor",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor/gpu:gpu_activation_header",
        "//xla/stream_executor/gpu:gpu_event",
        "//xla/stream_executor/gpu:gpu_kernel_header",
        "//xla/stream_executor/gpu:gpu_command_buffer",
        "//xla/stream_executor/gpu:gpu_runtime_header",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/stream_executor/gpu:gpu_timer",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
        "//xla/stream_executor/integrations:device_mem_allocator",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:fingerprint",
    ]),
    alwayslink = True,
)

cc_library(
    name = "rocm_kernel",
    srcs = if_rocm_is_configured(["rocm_kernel.cc"]),
    visibility = ["//visibility:public"],
    deps = if_rocm_is_configured([
        "//xla/stream_executor/gpu:gpu_kernel_header",
    ]),
    alwayslink = True,
)

rocm_library(
    name = "hip_conditional_kernels",
    srcs = if_rocm_is_configured(["hip_conditional_kernels.cu.cc"]),
    deps = if_rocm_is_configured(["@local_config_rocm//rocm:rocm_headers"]),
)

cc_library(
    name = "rocm_platform",
    srcs = if_rocm_is_configured(["rocm_platform.cc"]),
    hdrs = if_rocm_is_configured(["rocm_platform.h"]),
    visibility = ["//visibility:public"],
    deps = if_rocm_is_configured([
        ":rocm_driver",
        ":rocm_executor",
        ":rocm_platform_id",
        ":rocm_runtime",
        ":rocm_collectives",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "//xla/stream_executor",  # buildcleaner: keep
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor/platform",
    ]),
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
    deps = if_static([
        ":rocblas_if_rocm_configured",
    ]),
)

cc_library(
    name = "rocblas_if_rocm_configured",
    deps = if_rocm_is_configured([
        "@local_config_rocm//rocm:rocblas",
    ]),
)

cc_library(
    name = "rocblas_wrapper",
    srcs = if_rocm_is_configured(["rocblas_wrapper.h"]),
    hdrs = if_rocm_is_configured(["rocblas_wrapper.h"]),
    deps = if_rocm_is_configured([
        ":rocblas_if_static",
        ":rocm_executor",
        ":rocm_platform_id",
        "@local_config_rocm//rocm:rocm_headers",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
        "@local_tsl//tsl/platform:env",
        "//xla/tsl/util:determinism_for_kernels",
    ]),
    alwayslink = True,
)

cc_library(
    name = "rocblas_plugin",
    srcs = if_rocm_is_configured(["rocm_blas.cc"]),
    hdrs = if_rocm_is_configured(["rocm_blas.h"]),
    visibility = ["//visibility:public"],
    deps = if_rocm_is_configured([
        ":rocblas_if_static",
        ":rocblas_wrapper",
        ":hipblas_lt_header",
        ":rocm_executor",
        ":rocm_platform_id",
        "@eigen_archive//:eigen3",
        "//xla/stream_executor",
        "//xla/stream_executor:host_or_device_scalar",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor/gpu:gpu_activation",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_timer_header",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "//xla/stream_executor/platform",
        "//xla/stream_executor:blas",
        "//xla/stream_executor/platform:dso_loader",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_config_rocm//rocm:rocm_headers",
    ]),
    alwayslink = True,
)

cc_library(
    name = "hipfft_if_static",
    deps = if_static([
        ":hipfft_if_rocm_configured",
    ]),
)

cc_library(
    name = "hipfft_if_rocm_configured",
    deps = if_rocm_is_configured([
        "@local_config_rocm//rocm:hipfft",
    ]),
)

cc_library(
    name = "hipfft_plugin",
    srcs = if_rocm_is_configured(["rocm_fft.cc"]),
    hdrs = if_rocm_is_configured(["rocm_fft.h"]),
    visibility = ["//visibility:public"],
    deps = if_rocm_is_configured([
        ":hipfft_if_static",
        ":rocm_platform_id",
        "//xla/stream_executor",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor/gpu:gpu_activation",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_kernel_header",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:env",
    ]),
    alwayslink = True,
)

cc_library(
    name = "miopen_if_static",
    deps = if_static([
        ":miopen_if_rocm_configured",
    ]),
)

cc_library(
    name = "miopen_if_rocm_configured",
    deps = if_rocm_is_configured([
        "@local_config_rocm//rocm:miopen",
    ]),
)

cc_library(
    name = "miopen_plugin",
    srcs = if_rocm_is_configured(["rocm_dnn.cc"]),
    hdrs = if_rocm_is_configured(["rocm_dnn.h"]),
    copts = [
        # STREAM_EXECUTOR_CUDNN_WRAP would fail on Clang with the default
        # setting of template depth 256
        "-ftemplate-depth-512",
    ],
    visibility = ["//visibility:public"],
    deps = if_rocm_is_configured([
        ":miopen_if_static",
        ":rocm_diagnostics",
        ":rocm_driver",
        ":rocm_executor",
        ":rocm_platform_id",
        "@eigen_archive//:eigen3",
        "//xla/stream_executor",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor/gpu:gpu_activation_header",
        "//xla/stream_executor/gpu:gpu_stream_header",
        "//xla/stream_executor/gpu:gpu_timer_header",
        "//xla/stream_executor/gpu:gpu_types_header",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:env_impl",
        "//xla/tsl/util:env_var",
        "//xla/tsl/util:determinism_for_kernels",
    ]),
    alwayslink = True,
)

cc_library(
    name = "hiprand_if_static",
    deps = if_static([
        ":hiprand_if_rocm_configured",
    ]),
)

cc_library(
    name = "hiprand_if_rocm_configured",
    deps = if_rocm_is_configured([
        "@local_config_rocm//rocm:hiprand",
    ]),
)

cc_library(
    name = "hipsparse_if_static",
    deps = if_static([
        ":hipsparse_if_rocm_configured",
    ]),
)

cc_library(
    name = "hipsparse_if_rocm_configured",
    deps = if_rocm_is_configured([
        "@local_config_rocm//rocm:hipsparse",
    ]),
)

cc_library(
    name = "hipsparse_wrapper",
    srcs = if_rocm_is_configured(["hipsparse_wrapper.h"]),
    hdrs = if_rocm_is_configured(["hipsparse_wrapper.h"]),
    deps = if_rocm_is_configured([
        ":hipsparse_if_static",
        ":rocm_executor",
        ":rocm_platform_id",
        "@local_config_rocm//rocm:rocm_headers",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
        "@local_tsl//tsl/platform:env",
    ]),
    alwayslink = True,
)

cc_library(
    name = "rocsolver_if_static",
    deps = if_static([
        ":rocsolver_if_rocm_configured",
    ]),
)

cc_library(
    name = "rocsolver_if_rocm_configured",
    deps = if_rocm_is_configured([
        "@local_config_rocm//rocm:rocsolver",
    ]),
)

cc_library(
    name = "rocsolver_wrapper",
    srcs = if_rocm_is_configured(["rocsolver_wrapper.h"]),
    hdrs = if_rocm_is_configured(["rocsolver_wrapper.h"]),
    deps = if_rocm_is_configured([
        ":rocm_executor",
        ":rocm_platform_id",
        ":rocsolver_if_static",
        "@local_config_rocm//rocm:rocm_headers",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
        "@local_tsl//tsl/platform:env",
    ]),
    alwayslink = True,
)

cc_library(
    name = "hipsolver_if_static",
    deps = if_static([
        ":hipsolver_if_rocm_configured",
    ]),
)

cc_library(
    name = "hipsolver_if_rocm_configured",
    deps = if_rocm_is_configured([
        "@local_config_rocm//rocm:hipsolver",
    ]),
)

cc_library(
    name = "hipsolver_wrapper",
    srcs = if_rocm_is_configured(["hipsolver_wrapper.h"]),
    hdrs = if_rocm_is_configured(["hipsolver_wrapper.h"]),
    deps = if_rocm_is_configured([
        ":rocm_executor",
        ":rocm_platform_id",
        ":hipsolver_if_static",
        "@local_config_rocm//rocm:rocm_headers",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
        "@local_tsl//tsl/platform:env",
    ]),
    alwayslink = True,
)

cc_library(
    name = "hipblaslt_if_static",
    deps = if_rocm_hipblaslt([
        "@local_config_rocm//rocm:hipblaslt",
    ]),
)

cc_library(
    name = "amdhipblaslt_plugin",
    srcs = if_rocm_is_configured(["hip_blas_lt.cc"]),
    hdrs = if_rocm_is_configured([
        "hip_blas_lt.h",
        "hipblaslt_wrapper.h",
        "hip_blas_utils.h",
    ]),
    deps = if_rocm_is_configured([
        ":rocm_executor",
        ":rocm_platform_id",
        ":rocblas_plugin",
        ":hip_blas_utils",
        ":hipblas_lt_header",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:errors",
        "//xla:status",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_helpers_header",
        "//xla/stream_executor/gpu:gpu_blas_lt",
        "//xla:util",
    ]) + if_static([
        ":hipblaslt_if_static",
    ]),
    alwayslink = True,
)

cc_library(
    name = "hipblas_lt_header",
    hdrs = if_rocm_is_configured([
        "hip_blas_lt.h",
        "hipblaslt_wrapper.h",
        "hip_blas_utils.h",
    ]),
    visibility = ["//visibility:public"],
    deps = if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:errors",
        "//xla:status",
        "//xla/stream_executor:host_or_device_scalar",
        "//xla/stream_executor",
        "//xla/stream_executor/platform",
    ]),
)

cc_library(
    name = "hip_blas_utils",
    srcs = if_rocm_is_configured(["hip_blas_utils.cc"]),
    hdrs = if_rocm_is_configured(["hip_blas_utils.h"]),
    deps = if_rocm_is_configured([
        ":rocblas_plugin",
        ":hipblas_lt_header",
        "@com_google_absl//absl/strings",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:errors",
        "//xla/stream_executor",
    ]),
)

cc_library(
    name = "roctracer_if_static",
    deps = if_static([
        ":roctracer_if_rocm_configured",
    ]),
)

cc_library(
    name = "roctracer_if_rocm_configured",
    deps = if_rocm_is_configured([
        "@local_config_rocm//rocm:roctracer",
    ]),
)

cc_library(
    name = "roctracer_wrapper",
    srcs = if_rocm_is_configured(["roctracer_wrapper.h"]),
    hdrs = if_rocm_is_configured(["roctracer_wrapper.h"]),
    deps = if_rocm_is_configured([
        ":rocm_executor",
        ":rocm_platform_id",
        ":roctracer_if_static",
        "@local_config_rocm//rocm:rocm_headers",
        "//xla/stream_executor/platform",
        "//xla/stream_executor/platform:dso_loader",
        "@local_tsl//tsl/platform:env",
    ]),
    alwayslink = True,
)

rocm_library(
    name = "rocm_helpers",
    srcs = if_rocm_is_configured(["rocm_helpers.cu.cc"]),
    deps = if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
    alwayslink = True,
)

cc_library(
    name = "all_runtime",
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = if_rocm_is_configured([
        ":miopen_plugin",
        ":hipfft_plugin",
        ":rocblas_plugin",
        ":rocm_driver",
        ":rocm_platform",
        ":rocm_helpers",
        ":amdhipblaslt_plugin",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "rocm_rpath",
    data = [],
    linkopts = select({
        "//conditions:default": [
            "-Wl,-rpath,../local_config_rocm/rocm/rocm/lib",
        ],
    }),
    deps = [],
)

cc_library(
    name = "stream_executor_rocm",
    deps = [
        ":rocm_rpath",
        "//xla/stream_executor:stream_executor_bundle",
    ] + if_static(
        [":all_runtime"],
    ),
)

# copybara:comment_begin(oss-only)
rocm_embedded_test_modules(
    name = "add_i32_kernel",
    srcs = if_rocm_is_configured(["add_i32_kernel.cu.cc"]),
)
# copybara:comment_end
