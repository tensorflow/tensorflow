# Description:
# Platform-specific build configurations.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_copts", "tf_cuda_library")
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "gtest",
    testonly = 1,
    copts = tf_copts(),
    deps = [
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "tensorflow_platform_specific",
    copts = tf_copts(),
    linkstatic = 1,
    deps = [],
)

cc_library(
    name = "_empty_lib",
    visibility = ["//visibility:private"],
)

alias(
    name = "_cuda_runtime",
    actual = select({
        "//tensorflow:framework_shared_object": ":_empty_lib",
        "//conditions:default": "//tensorflow/compiler/xla/stream_executor/cuda:all_runtime",
    }),
    visibility = ["//visibility:private"],
)

tf_cuda_library(
    name = "stream_executor",
    cuda_deps = [
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_activation",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_activation",
    ],
    deps = [
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:dnn",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:multi_platform_manager",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform_id",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_platform_id",
    ] + select({
        "//tensorflow:macos": ["IOKit"],
        "//conditions:default": [],
    }) + select({
        "//tensorflow:is_cuda_enabled": [":_cuda_runtime"],
        "//tensorflow:using_rocm_hipcc": [
            "//tensorflow/compiler/xla/stream_executor/rocm:all_runtime",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "stream_executor_cuda",
    deps = [
        ":stream_executor_no_cuda",
    ] + select({
        "//tensorflow:oss": ["//tensorflow/compiler/xla/stream_executor/cuda:cudart_stub"],
        "//conditions:default": [
            "//tensorflow/compiler/xla/stream_executor/cuda:all_runtime",
            ":cuda",
        ],
    }) + select({
        "//tensorflow:macos": ["IOKit"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "stream_executor_rocm",
    deps = [
        ":stream_executor_no_cuda",
        ":rocm",
    ] + if_static(
        ["//tensorflow/compiler/xla/stream_executor/rocm:all_runtime"],
    ) + select({
        "//conditions:default": [],
    }),
)

cc_library(
    name = "stream_executor_no_cuda",
    deps = [
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:dnn",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:multi_platform_manager",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform_id",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_platform_id",
    ],
)

# Dummy stream executor cuda plugins.
cc_library(
    name = "cublas_plugin",
    srcs = [],
)

cc_library(
    name = "cufft_plugin",
    srcs = [],
)

cc_library(
    name = "cudnn_plugin",
    srcs = [],
)

# OSX framework for device driver access
cc_library(
    name = "IOKit",
    linkopts = ["-framework IOKit"],
)

# Minimal lib so that tools used for mobile compilation
# don't have to depend on platformlib.
cc_library(
    name = "proto_parsing",
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:protos_all_cc",
    ],
)

# Minimal lib to be used by tensorflow/core:framework_lite.
# This provides minimal support for writing operator implementations (kernels),
# and excludes anything that can bloat binary size if used.
cc_library(
    name = "minimal",
    srcs = [],
    copts = tf_copts(),
)

cc_library(
    name = "platformlib",
    copts = tf_copts(),
    deps = [
        ":gif",
        ":jpeg",
        "//tensorflow/core:protos_all_cc",
        "@com_googlesource_code_re2//:re2",
        "@farmhash_archive//:farmhash",
        "@fft2d",
        "@highwayhash//:sip_hash",
        "@zlib",
    ],
)

cc_library(
    name = "gif",
    copts = tf_copts(),
    deps = [
        "@gif",
    ],
)

cc_library(
    name = "jpeg",
    copts = tf_copts(),
    deps = [
        "@libjpeg_turbo//:jpeg",
    ],
)

cc_library(
    name = "png",
    copts = tf_copts(),
    deps = [
        "@png",
        "@zlib",
    ],
)

cc_library(
    name = "test_main",
    testonly = 1,
    linkstatic = 1,
    deps = [],
)

cc_library(
    name = "cuda",
    data = [
        "@local_config_cuda//cuda:cudart",
    ],
    linkopts = select({
        "//tensorflow:macos": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib",
        ],
        "//conditions:default": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib64",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib64",
        ],
    }),
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_library(
    name = "rocm",
    data = [],
    linkopts = select({
        "//conditions:default": [
            "-Wl,-rpath,../local_config_rocm/rocm/rocm/lib",
        ],
    }),
    deps = [],
)
