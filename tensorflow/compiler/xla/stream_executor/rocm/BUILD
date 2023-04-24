# Description:
#   ROCm-platform specific StreamExecutor support code.

load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/compiler/xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load("//tensorflow/tsl:tsl.bzl", "set_external_visibility", "tsl_copts")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured", "rocm_copts")
load("//tensorflow/tsl/platform:build_config_root.bzl", "if_static")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = set_external_visibility([":friends"]),
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
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_diagnostics_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:platform_port",
    ]),
)

cc_library(
    name = "rocm_driver",
    srcs = if_rocm_is_configured(["rocm_driver.cc"]),
    hdrs = if_rocm_is_configured(["rocm_driver_wrapper.h"]),
    deps = if_rocm_is_configured([
        ":rocm_diagnostics",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "//tensorflow/compiler/xla/stream_executor:device_options",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_driver_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:numbers",
        "//tensorflow/tsl/platform:stacktrace",
        "//tensorflow/tsl/platform:static_threadlocal",
    ]),
)

cc_library(
    name = "rocm_activation",
    srcs = [],
    hdrs = if_rocm_is_configured(["rocm_activation.h"]),
    deps = if_rocm_is_configured([
        ":rocm_driver",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_internal",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_activation",
        "//tensorflow/compiler/xla/stream_executor/platform",
    ]),
)

cc_library(
    name = "rocm_event",
    srcs = if_rocm_is_configured(["rocm_event.cc"]),
    deps = if_rocm_is_configured([
        ":rocm_driver",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_event_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_executor_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream_header",
    ]),
)

cc_library(
    name = "rocm_gpu_executor",
    srcs = if_rocm_is_configured(["rocm_gpu_executor.cc"]),
    deps = if_rocm_is_configured([
        ":rocm_diagnostics",
        ":rocm_driver",
        ":rocm_event",
        ":rocm_kernel",
        ":rocm_platform_id",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:plugin_registry",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_internal",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/compiler/xla/stream_executor:timer",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_activation_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_event",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_kernel_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_timer",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
    ]),
    alwayslink = True,
)

cc_library(
    name = "rocm_kernel",
    srcs = if_rocm_is_configured(["rocm_kernel.cc"]),
    visibility = ["//visibility:public"],
    deps = if_rocm_is_configured([
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_kernel_header",
    ]),
    alwayslink = True,
)

cc_library(
    name = "rocm_platform",
    srcs = if_rocm_is_configured(["rocm_platform.cc"]),
    hdrs = if_rocm_is_configured(["rocm_platform.h"]),
    visibility = ["//visibility:public"],
    deps = if_rocm_is_configured([
        ":rocm_driver",
        ":rocm_gpu_executor",
        ":rocm_platform_id",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "//tensorflow/compiler/xla/stream_executor",  # buildcleaner: keep
        "//tensorflow/compiler/xla/stream_executor:executor_cache",
        "//tensorflow/compiler/xla/stream_executor:multi_platform_manager",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
    ]),
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "rocm_platform_id",
    srcs = ["rocm_platform_id.cc"],
    hdrs = ["rocm_platform_id.h"],
    deps = ["//tensorflow/compiler/xla/stream_executor:platform"],
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
        ":rocm_gpu_executor",
        ":rocm_platform_id",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/util:determinism_for_kernels",
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
        ":rocm_gpu_executor",
        ":rocm_helpers",
        ":rocm_platform_id",
        "//third_party/eigen3",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:host_or_device_scalar",
        "//tensorflow/compiler/xla/stream_executor:plugin_registry",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
        "//tensorflow/compiler/xla/stream_executor:timer",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_activation",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_helpers_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_timer_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/stream_executor:blas",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
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
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:fft",
        "//tensorflow/compiler/xla/stream_executor:plugin_registry",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_activation",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_helpers_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_executor_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_kernel_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/tsl/platform:env",
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
        ":rocm_gpu_executor",
        ":rocm_platform_id",
        "//third_party/eigen3",
        "//tensorflow/compiler/xla/stream_executor:dnn",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:plugin_registry",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/compiler/xla/stream_executor:temporary_device_memory",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_activation_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_timer_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_types_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/util:env_var",
        "//tensorflow/tsl/util:determinism_for_kernels",
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
    name = "rocrand_plugin",
    srcs = if_rocm_is_configured(["rocm_rng.cc"]),
    hdrs = if_rocm_is_configured([]),
    deps = if_rocm_is_configured([
        ":hiprand_if_static",
        ":rocm_gpu_executor",
        ":rocm_platform_id",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:plugin_registry",
        "//tensorflow/compiler/xla/stream_executor:rng",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_activation_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_helpers_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_executor_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_rng_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
    ]),
    alwayslink = True,
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
        ":rocm_gpu_executor",
        ":rocm_platform_id",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
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
        ":rocm_gpu_executor",
        ":rocm_platform_id",
        ":rocsolver_if_static",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
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
        ":rocm_gpu_executor",
        ":rocm_platform_id",
        ":hipsolver_if_static",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
    ]),
    alwayslink = True,
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
        ":rocm_gpu_executor",
        ":rocm_platform_id",
        ":roctracer_if_static",
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
    ]),
    alwayslink = True,
)

cc_library(
    name = "rocm_helpers",
    srcs = if_rocm_is_configured(["rocm_helpers.cu.cc"]),
    copts = rocm_copts(),
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
        ":rocrand_plugin",
        ":rocm_driver",
        ":rocm_platform",
        ":rocm_helpers",
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
        "//tensorflow/compiler/xla/stream_executor:stream_executor_bundle",
    ] + if_static(
        [":all_runtime"],
    ),
)
