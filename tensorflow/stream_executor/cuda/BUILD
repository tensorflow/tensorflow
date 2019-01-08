# Description:
#   CUDA-platform specific StreamExecutor support code.

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "//tensorflow/stream_executor:build_defs.bzl",
    "stream_executor_friends",
    "tf_additional_cuda_driver_deps",
    "tf_additional_cuda_platform_deps",
    "tf_additional_cudnn_plugin_deps",
)
load("//tensorflow:tensorflow.bzl", "tf_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

package(
    default_visibility = [":friends"],
)

# Filegroup used to collect source files for the dependency check.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "cuda_platform_id",
    srcs = ["cuda_platform_id.cc"],
    hdrs = ["cuda_platform_id.h"],
    deps = ["//tensorflow/stream_executor:platform"],
)

cc_library(
    name = "cuda_platform",
    srcs = ["cuda_platform.cc"],
    hdrs = ["cuda_platform.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_driver",
        ":cuda_gpu_executor",
        ":cuda_platform_id",
        "//tensorflow/stream_executor",  # buildcleaner: keep
        "//tensorflow/stream_executor:executor_cache",
        "//tensorflow/stream_executor:multi_platform_manager",
        "//tensorflow/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ] + tf_additional_cuda_platform_deps(),
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "cuda_diagnostics",
    srcs = ["cuda_diagnostics.cc"],
    hdrs = ["cuda_diagnostics.h"],
    deps = [
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cuda_driver",
    srcs = ["cuda_driver.cc"],
    hdrs = ["cuda_driver.h"],
    deps = [
        ":cuda_diagnostics",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/stream_executor:device_options",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ] + tf_additional_cuda_driver_deps(),
)

# The activation library is tightly coupled to the executor library.
# TODO(leary) split up cuda_gpu_executor.cc so that this can stand alone.
cc_library(
    name = "cuda_activation_header",
    hdrs = ["cuda_activation.h"],
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/stream_executor/platform"],
)

cc_library(
    name = "cuda_activation",
    srcs = ["cuda_activation.cc"],
    hdrs = ["cuda_activation.h"],
    deps = [
        ":cuda_driver",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor:stream_executor_internal",
        "//tensorflow/stream_executor/platform",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "cuda_gpu_executor_header",
    textual_hdrs = ["cuda_gpu_executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_kernel",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "cublas_plugin",
    srcs = ["cuda_blas.cc"],
    hdrs = ["cuda_blas.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_activation",
        ":cuda_gpu_executor",
        ":cuda_helpers",
        ":cuda_platform_id",
        ":cuda_stream",
        ":cuda_timer",
        "@com_google_absl//absl/strings",
        "//third_party/eigen3",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/core:lib_internal",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:host_or_device_scalar",
        "//tensorflow/stream_executor:plugin_registry",
        "//tensorflow/stream_executor:scratch_allocator",
        "//tensorflow/stream_executor:timer",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "//tensorflow/stream_executor/platform:dso_loader",
    ] + if_static(
        [
            "@local_config_cuda//cuda:cublas",
            "@local_config_cuda//cuda:cudart_static",
        ],
        ["@local_config_cuda//cuda:cudart"],
    ),
    alwayslink = True,
)

cc_library(
    name = "cufft_plugin",
    srcs = ["cuda_fft.cc"],
    hdrs = ["cuda_fft.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_activation_header",
        ":cuda_gpu_executor_header",
        ":cuda_helpers",
        ":cuda_platform_id",
        ":cuda_stream",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:fft",
        "//tensorflow/stream_executor:plugin_registry",
        "//tensorflow/stream_executor:scratch_allocator",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "//tensorflow/stream_executor/platform:dso_loader",
    ] + if_static(["@local_config_cuda//cuda:cufft"]),
    alwayslink = True,
)

cc_library(
    name = "cudnn_plugin",
    srcs = ["cuda_dnn.cc"],
    hdrs = ["cuda_dnn.h"],
    copts = [
        # STREAM_EXECUTOR_CUDNN_WRAP would fail on Clang with the default
        # setting of template depth 256
        "-ftemplate-depth-512",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_activation",
        ":cuda_diagnostics",
        ":cuda_driver",
        ":cuda_gpu_executor",
        ":cuda_platform_id",
        ":cuda_stream",
        ":cuda_timer",
        ":cudnn_version",
        "@com_google_absl//absl/strings",
        "//third_party/eigen3",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:logger",
        "//tensorflow/stream_executor:dnn",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:logging_proto_cc",
        "//tensorflow/stream_executor:plugin_registry",
        "//tensorflow/stream_executor:scratch_allocator",
        "//tensorflow/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/stream_executor:temporary_device_memory",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "//tensorflow/stream_executor/platform:dso_loader",
    ] + tf_additional_cudnn_plugin_deps() + if_static(
        [
            "@local_config_cuda//cuda:cudnn",
            "@local_config_cuda//cuda:cudart_static",
        ],
        ["@local_config_cuda//cuda:cudart"],
    ),
    alwayslink = True,
)

cc_library(
    name = "curand_plugin",
    srcs = ["cuda_rng.cc"],
    hdrs = ["cuda_rng.h"],
    deps = [
        ":cuda_activation",
        ":cuda_gpu_executor",
        ":cuda_helpers",
        ":cuda_platform_id",
        ":cuda_stream",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:plugin_registry",
        "//tensorflow/stream_executor:rng",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "//tensorflow/stream_executor/platform:dso_loader",
    ] + if_static(["@local_config_cuda//cuda:curand"]),
    alwayslink = True,
)

cc_library(
    name = "cuda_kernel",
    hdrs = ["cuda_kernel.h"],
    deps = [
        ":cuda_driver",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

# TODO(leary) we likely need to canonicalize/eliminate this.
cc_library(
    name = "cuda_helpers",
    textual_hdrs = ["cuda_helpers.h"],
)

cc_library(
    name = "cuda_event",
    srcs = ["cuda_event.cc"],
    hdrs = ["cuda_event.h"],
    deps = [
        ":cuda_driver",
        ":cuda_gpu_executor_header",
        ":cuda_stream",
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor/lib",
    ],
)

cc_library(
    name = "cuda_stream",
    srcs = ["cuda_stream.cc"],
    hdrs = ["cuda_stream.h"],
    deps = [
        ":cuda_driver",
        ":cuda_gpu_executor_header",
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor:stream_header",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
    ],
)

cc_library(
    name = "cuda_timer",
    srcs = ["cuda_timer.cc"],
    hdrs = ["cuda_timer.h"],
    deps = [
        ":cuda_driver",
        ":cuda_gpu_executor_header",
        ":cuda_stream",
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor/lib",
    ],
)

# It implements :cuda_gpu_executor_header
cc_library(
    name = "cuda_gpu_executor",
    srcs = ["cuda_gpu_executor.cc"],
    hdrs = ["cuda_gpu_executor.h"],
    deps = [
        ":cuda_activation",
        ":cuda_diagnostics",
        ":cuda_driver",
        ":cuda_event",
        ":cuda_kernel",
        ":cuda_platform_id",
        ":cuda_stream",
        ":cuda_timer",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:plugin_registry",
        "//tensorflow/stream_executor:stream_executor_internal",
        "//tensorflow/stream_executor:stream_executor_pimpl_header",
        "//tensorflow/stream_executor:timer",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = True,
)

cc_library(
    name = "cudnn_version",
    srcs = ["cudnn_version.cc"],
    hdrs = ["cudnn_version.h"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "cudnn_version_test",
    srcs = ["cudnn_version_test.cc"],
    deps = [
        ":cudnn_version",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "all_runtime",
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":cudnn_plugin",
        ":cufft_plugin",
        ":cublas_plugin",
        ":curand_plugin",
        ":cuda_driver",
        ":cuda_platform",
    ]),
    alwayslink = 1,
)
