# Description:
#   CUDA-platform specific StreamExecutor support code.

load("//tensorflow/tsl:tsl.bzl", "if_google", "tsl_copts")
load("//tensorflow/tsl:tsl.default.bzl", "tsl_gpu_cc_test")
load(
    "//tensorflow/compiler/xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
    "tf_additional_cuda_driver_deps",
    "tf_additional_cuda_platform_deps",
    "tf_additional_cudnn_plugin_copts",
    "tf_additional_cudnn_plugin_deps",
    "tf_additional_gpu_compilation_copts",
)
load(
    "//tensorflow/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "//tensorflow/tsl/platform:build_config_root.bzl",
    "if_static",
    "tf_cuda_tests_tags",
)
load(
    "//tensorflow/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "cuda_platform_id",
    srcs = ["cuda_platform_id.cc"],
    hdrs = ["cuda_platform_id.h"],
    deps = ["//tensorflow/compiler/xla/stream_executor:platform"],
)

cc_library(
    name = "cuda_platform",
    srcs = if_cuda_is_configured(["cuda_platform.cc"]),
    hdrs = if_cuda_is_configured(["cuda_platform.h"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured(
        [
            ":cuda_driver",
            ":cuda_gpu_executor",
            ":cuda_platform_id",
            ":cuda_activation",
            "//tensorflow/compiler/xla/stream_executor",  # buildcleaner: keep
            "//tensorflow/compiler/xla/stream_executor:executor_cache",
            "//tensorflow/compiler/xla/stream_executor:multi_platform_manager",
            "//tensorflow/compiler/xla/stream_executor:stream_executor_pimpl_header",
            "//tensorflow/compiler/xla/stream_executor/platform",
        ],
    ) + tf_additional_cuda_platform_deps() + [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "cuda_diagnostics",
    srcs = if_cuda_is_configured(["cuda_diagnostics.cc"]),
    hdrs = if_cuda_is_configured(["cuda_diagnostics.h"]),
    deps = if_cuda_is_configured([
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_diagnostics_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:platform_port",
    ]) + ["@com_google_absl//absl/strings:str_format"],
)

# Buildozer can not remove dependencies inside select guards, so we have to use
# an intermediate target.
cc_library(name = "ptxas_wrapper")

cc_library(name = "nvlink_wrapper")

# Buildozer can not remove dependencies inside select guards, so we have to use
# an intermediate target.
cc_library(name = "fatbinary_wrapper")

cc_library(
    name = "cuda_driver",
    srcs = if_cuda_is_configured(["cuda_driver.cc"]),
    hdrs = if_cuda_is_configured(["cuda_driver.h"]),
    deps = if_cuda_is_configured([
        ":cuda_diagnostics",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/compiler/xla/stream_executor:device_options",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_driver_header",
        "//tensorflow/compiler/xla/stream_executor:platform",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:static_threadlocal",
        "//tensorflow/tsl/util:env_var",
    ] + tf_additional_cuda_driver_deps()) + select({
        # include dynamic loading implementation only when if_cuda_is_configured and build dynamically
        "//tensorflow/tsl:is_cuda_enabled_and_oss": ["cudart_stub"],
        "//conditions:default": ["//tensorflow/tsl/platform:cuda"],
    }) + [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

tsl_gpu_cc_test(
    name = "stream_search_test",
    size = "small",
    srcs = ["stream_search_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_impl",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

tsl_gpu_cc_test(
    name = "cuda_driver_test",
    srcs = ["cuda_driver_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # TODO(b/171512140): re-enable.
    ],
    deps = [
        ":cuda_driver",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

tsl_gpu_cc_test(
    name = "memcpy_test",
    srcs = ["memcpy_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # TODO(b/171512140): re-enable.
    ],
    deps = [
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "//tensorflow/compiler/xla/stream_executor:multi_platform_manager",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_impl",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

alias(
    name = "cudart_stub",
    actual = "//tensorflow/tsl/cuda:cudart_stub",
    visibility = ["//visibility:public"],
)

# The activation library is tightly coupled to the executor library.
# TODO(leary) split up cuda_gpu_executor.cc so that this can stand alone.
cc_library(
    name = "cuda_activation_header",
    hdrs = ["cuda_activation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_activation_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
    ],
)

cc_library(
    name = "cuda_activation",
    srcs = [],
    hdrs = if_cuda_is_configured(["cuda_activation.h"]),
    deps = if_cuda_is_configured([
        ":cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_internal",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_activation",
        "//tensorflow/compiler/xla/stream_executor/platform",
    ]),
)

cc_library(
    name = "cuda_gpu_executor_header",
    textual_hdrs = if_cuda_is_configured(["cuda_gpu_executor.h"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":cuda_kernel",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_executor_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
    ]),
)

alias(
    name = "cublas_stub",
    actual = "//tensorflow/tsl/cuda:cublas_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "cublas_lib",
    actual = "//tensorflow/tsl/cuda:cublas_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cublas_lt_header",
    hdrs = if_cuda_is_configured([
        "cuda_blas_lt.h",
        "cuda_blas_utils.h",
    ]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/compiler/xla/stream_executor:host_or_device_scalar",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/platform",
    ]) + ["//tensorflow/tsl/platform:errors"],
)

alias(
    name = "cublas_lt_stub",
    actual = "//tensorflow/tsl/cuda:cublas_lt_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "cublas_lt_lib",
    actual = "//tensorflow/tsl/cuda:cublas_lt_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cublas_plugin",
    srcs = if_cuda_is_configured([
        "cuda_blas.cc",
        "cuda_blas_lt.cc",
    ]),
    hdrs = if_cuda_is_configured([
        "cuda_blas.h",
        "cuda_blas_lt.h",
    ]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":cublas_lib",
        ":cublas_lt_lib",
        ":cuda_activation",
        ":cuda_blas_utils",
        ":cuda_gpu_executor",
        ":cuda_helpers",
        ":cuda_platform_id",
        ":cuda_stream",
        ":cuda_timer",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "//third_party/eigen3",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "//tensorflow/compiler/xla/stream_executor:host_or_device_scalar",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_activation_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_executor_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_helpers_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_timer_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_types_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:tensor_float_32_hdr_lib",
    ]) + if_static([
        "//tensorflow/tsl/platform:tensor_float_32_utils",
    ]),
    alwayslink = True,
)

cc_library(
    name = "cuda_blas_utils",
    srcs = if_cuda_is_configured(["cuda_blas_utils.cc"]),
    hdrs = if_cuda_is_configured(["cuda_blas_utils.h"]),
    deps = if_cuda_is_configured([
        ":cublas_lib",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
    ]) + ["//tensorflow/tsl/platform:errors"],
)

alias(
    name = "cufft_stub",
    actual = "//tensorflow/tsl/cuda:cufft_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "cufft_lib",
    actual = "//tensorflow/tsl/cuda:cufft_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cufft_plugin",
    srcs = if_cuda_is_configured(["cuda_fft.cc"]),
    hdrs = if_cuda_is_configured(["cuda_fft.h"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":cuda_activation_header",
        ":cuda_gpu_executor_header",
        ":cuda_platform_id",
        ":cuda_stream",
        ":cuda_helpers",
        ":cufft_lib",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:fft",
        "//tensorflow/compiler/xla/stream_executor:plugin_registry",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_helpers_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
    ]) + ["//tensorflow/tsl/platform:errors"],
    alwayslink = True,
)

alias(
    name = "cudnn_stub",
    actual = "//tensorflow/tsl/cuda:cudnn_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "cudnn_lib",
    actual = "//tensorflow/tsl/cuda:cudnn_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_dnn_headers",
    textual_hdrs = ["cuda_dnn.h"],
    deps = if_cuda_is_configured([
        ":cuda_activation_header",
        "//tensorflow/compiler/xla/stream_executor:dnn",
        "//tensorflow/compiler/xla/stream_executor:plugin_registry",
    ]),
)

cc_library(
    name = "cudnn_plugin",
    srcs = if_cuda_is_configured(["cuda_dnn.cc"]),
    hdrs = if_cuda_is_configured(["cuda_dnn.h"]),
    copts = tf_additional_cudnn_plugin_copts(),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":cuda_activation",
        ":cuda_diagnostics",
        ":cuda_driver",
        ":cuda_gpu_executor",
        ":cuda_platform_id",
        ":cuda_stream",
        ":cuda_timer",
        ":cudnn_lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@cudnn_frontend_archive//:cudnn_frontend",
        "//third_party/eigen3",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudnn_header",
        "//tensorflow/tsl/cuda:cudnn_version",
        "//tensorflow/tsl/platform:tensor_float_32_utils",
        "//tensorflow/tsl/util:determinism",
        "//tensorflow/compiler/xla/stream_executor:dnn",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:plugin_registry",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/compiler/xla/stream_executor:temporary_device_memory",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:tensor_float_32_hdr_lib",
        "//tensorflow/tsl/util:env_var",
    ]) + tf_additional_cudnn_plugin_deps() + [
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = True,
)

alias(
    name = "curand_stub",
    actual = "//tensorflow/tsl/cuda:curand_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "curand_lib",
    actual = "//tensorflow/tsl/cuda:curand_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "curand_plugin",
    srcs = if_cuda_is_configured(["cuda_rng.cc"]),
    hdrs = if_cuda_is_configured(["cuda_rng.h"]),
    deps = if_cuda_is_configured([
        ":cuda_activation",
        ":cuda_gpu_executor",
        ":cuda_platform_id",
        ":cuda_stream",
        ":cuda_helpers",
        ":curand_lib",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:plugin_registry",
        "//tensorflow/compiler/xla/stream_executor:rng",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_helpers_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_rng_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
    ]),
    alwayslink = True,
)

alias(
    name = "cupti_stub",
    actual = "//tensorflow/tsl/cuda:cupti_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "cusolver_stub",
    actual = "//tensorflow/tsl/cuda:cusolver_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "cusolver_lib",
    actual = "//tensorflow/tsl/cuda:cusolver_lib",
    visibility = ["//visibility:public"],
)

alias(
    name = "cusparse_stub",
    actual = "//tensorflow/tsl/cuda:cusparse_stub",
    visibility = ["//visibility:public"],
)

alias(
    name = "cusparse_lib",
    actual = "//tensorflow/tsl/cuda:cusparse_lib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_kernel",
    srcs = if_cuda_is_configured(["cuda_kernel.cc"]),
    hdrs = if_cuda_is_configured(["cuda_kernel.h"]),
    deps = if_cuda_is_configured([
        ":cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_kernel_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
    ]),
)

# TODO(leary) we likely need to canonicalize/eliminate this.
cc_library(
    name = "cuda_helpers",
    textual_hdrs = if_cuda_is_configured(["cuda_helpers.h"]),
    deps = if_cuda_is_configured([
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_helpers_header",
    ]),
)

cc_library(
    name = "cuda_event",
    srcs = if_cuda_is_configured(["cuda_event.cc"]),
    hdrs = if_cuda_is_configured(["cuda_event.h"]),
    deps = if_cuda_is_configured([
        ":cuda_driver",
        ":cuda_gpu_executor_header",
        ":cuda_stream",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_event",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream_header",
    ]),
)

cc_library(
    name = "cuda_graph",
    srcs = if_cuda_is_configured(["cuda_graph.cc"]),
    hdrs = if_cuda_is_configured(["cuda_graph.h"]),
    deps = if_cuda_is_configured([
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream",
    ]),
)

cc_library(
    name = "cuda_stream",
    srcs = [],
    hdrs = if_cuda_is_configured(["cuda_stream.h"]),
    deps = if_cuda_is_configured([
        ":cuda_driver",
        ":cuda_gpu_executor_header",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream",
        "//tensorflow/compiler/xla/stream_executor/platform",
    ]),
)

cc_library(
    name = "cuda_timer",
    srcs = [],
    hdrs = if_cuda_is_configured(["cuda_timer.h"]),
    deps = if_cuda_is_configured([
        ":cuda_driver",
        ":cuda_gpu_executor_header",
        ":cuda_stream",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_timer",
    ]),
)

cc_library(
    name = "cuda_asm_compiler",
    srcs = if_cuda_is_configured(["cuda_asm_compiler.cc"]),
    copts = tf_additional_gpu_compilation_copts(),
    deps = if_cuda_is_configured([
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings:str_format",
        "//tensorflow/compiler/xla/stream_executor/gpu:asm_compiler_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_diagnostics_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_driver_header",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/tsl/platform:errors",
    ]),
)

cc_library(
    name = "cuda_gpu_executor",
    srcs = if_cuda_is_configured(["cuda_gpu_executor.cc"]),
    hdrs = if_cuda_is_configured(["cuda_gpu_executor.h"]),
    deps = if_cuda_is_configured([
        ":cuda_activation",
        ":cuda_diagnostics",
        ":cuda_driver",
        ":cuda_event",
        ":cuda_kernel",
        ":cuda_platform_id",
        ":cuda_stream",
        ":cuda_timer",
        ":cuda_asm_compiler",
        "@com_google_absl//absl/strings",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:plugin_registry",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_internal",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/compiler/xla/stream_executor:timer",
        "//tensorflow/compiler/xla/stream_executor/gpu:asm_compiler",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_executor_header",
        "//tensorflow/compiler/xla/stream_executor/platform",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
    ]) + [
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:numbers",
        "//tensorflow/tsl/platform:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "all_runtime",
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cublas_plugin",
        ":cuda_driver",
        ":cuda_platform",
        ":cudnn_plugin",
        ":cufft_plugin",
        ":curand_plugin",
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
    deps = [
        "//tensorflow/compiler/xla/stream_executor:stream_executor_bundle",
    ] + if_google(
        select({
            # copybara:uncomment_begin(different config setting in OSS)
            # "//tools/cc_target_os:gce": [],
            # copybara:uncomment_end_and_comment_begin
            "//conditions:default": [
                "@local_config_cuda//cuda:cudart_static",  # buildcleaner: keep
                ":cuda_platform",
            ],
        }),
        [
            ":cudart_stub",
        ] + select({
            "//tensorflow/tsl:macos": ["IOKit"],
            "//conditions:default": [],
        }),
    ),
)

tsl_gpu_cc_test(
    name = "redzone_allocator_test",
    srcs = ["redzone_allocator_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # TODO(b/171512140): re-enable.
    ],
    deps = [
        ":cuda_activation",
        ":cuda_gpu_executor",
        ":stream_executor_cuda",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:device_memory_allocator",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:kernel",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_impl",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_asm_opts",
        "//tensorflow/compiler/xla/stream_executor/gpu:redzone_allocator",
        "//tensorflow/tsl/framework:allocator",
        "//tensorflow/tsl/framework:allocator_registry_impl",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/profiler/backends/cpu:traceme_recorder_impl",
        "//tensorflow/tsl/profiler/utils:time_utils_impl",
    ],
)
