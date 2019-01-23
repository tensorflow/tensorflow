# Description:
# Platform-specific build configurations.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "if_cuda")
load("//tensorflow:tensorflow.bzl", "tf_copts")
load("//tensorflow:tensorflow.bzl", "tf_cuda_library")
load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")
load("@local_config_sycl//sycl:platform.bzl", "sycl_library_path")
load("@local_config_sycl//sycl:build_defs.bzl", "if_ccpp")

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

tf_cuda_library(
    name = "stream_executor",
    cuda_deps = ["//tensorflow/stream_executor/cuda:cuda_activation"],
    deps = [
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor:dnn",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:multi_platform_manager",
        "//tensorflow/stream_executor:scratch_allocator",
        "//tensorflow/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/stream_executor/host:host_platform_id",
        "//tensorflow/stream_executor/platform:dso_loader",
    ] + select({
        "@local_config_cuda//cuda:darwin": ["IOKit"],
        "//conditions:default": [],
    }) + select({
        "//tensorflow:using_cuda_clang": ["//tensorflow/stream_executor/cuda:all_runtime"],
        "//tensorflow:using_cuda_nvcc": ["//tensorflow/stream_executor/cuda:all_runtime"],
        "//tensorflow:using_cuda_clang_with_dynamic_build": [],
        "//tensorflow:using_cuda_nvcc_with_dynamic_build": [],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "stream_executor_cuda",
    deps = [
        ":stream_executor_no_cuda",
        ":cuda",
    ] + if_static(
        ["//tensorflow/stream_executor/cuda:all_runtime"],
    ) + select({
        "@local_config_cuda//cuda:darwin": ["IOKit"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "stream_executor_no_cuda",
    deps = [
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor:dnn",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:multi_platform_manager",
        "//tensorflow/stream_executor:scratch_allocator",
        "//tensorflow/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/stream_executor/host:host_platform",
        "//tensorflow/stream_executor/host:host_platform_id",
        "//tensorflow/stream_executor/platform:dso_loader",
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
        "//tensorflow/core:protos_cc",
    ],
)

# Minimal lib so that tools used for mobile compilation
# don't have to depend on platformlib.
cc_library(
    name = "logging",
    copts = tf_copts(),
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
    name = "base",
    srcs = [],
    copts = tf_copts(),
)

cc_library(
    name = "port",
    srcs = [],
    copts = tf_copts(),
)

cc_library(
    name = "protobuf",
    srcs = [],
    copts = tf_copts(),
)

cc_library(
    name = "env",
    srcs = [],
    copts = tf_copts(),
)

cc_library(
    name = "other",
    srcs = [],
    copts = tf_copts(),
    deps = [
        "@com_googlesource_code_re2//:re2",
        "@farmhash_archive//:farmhash",
        "@fft2d",
        "@highwayhash//:sip_hash",
    ],
)

cc_library(
    name = "platformlib",
    copts = tf_copts(),
    deps = [
        ":gif",
        ":jpeg",
        "//tensorflow/core:protos_cc",
        "@com_googlesource_code_re2//:re2",
        "@farmhash_archive//:farmhash",
        "@fft2d",
        "@highwayhash//:sip_hash",
        "@zlib_archive//:zlib",
    ],
)

cc_library(
    name = "stacktrace",
    srcs = [],
)

cc_library(
    name = "gif",
    copts = tf_copts(),
    deps = [
        "@gif_archive//:gif",
    ],
)

cc_library(
    name = "jpeg",
    copts = tf_copts(),
    deps = [
        "@jpeg",
    ],
)

cc_library(
    name = "png",
    copts = tf_copts(),
    deps = [
        "@png_archive//:png",
        "@zlib_archive//:zlib",
    ],
)

cc_library(
    name = "protos_cc_impl",
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:protos_all_cc_impl",
    ],
)

cc_library(
    name = "protos_cc",
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "test_lite_main",
    testonly = 1,
    linkstatic = 1,
    deps = [],
)

cc_library(
    name = "test_main",
    testonly = 1,
    linkstatic = 1,
    deps = [],
)

filegroup(
    name = "android_proto_lib_portable_proto",
    srcs = [],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda",
    data = [
        "@local_config_cuda//cuda:cudart",
    ],
    linkopts = select({
        "@local_config_cuda//cuda:darwin": [
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
    name = "sycl",
    data = if_ccpp([
        "@local_config_sycl//sycl:{}".format(sycl_library_path("ComputeCpp")),
    ]),
    linkopts = if_ccpp([
        "-Wl,-rpath,../local_config_sycl/sycl/lib",
    ]),
    deps = if_ccpp(
        ["@local_config_sycl//sycl:syclrt"],
        ["@local_config_sycl//sycl:sycl_headers"],
    ),
)

filegroup(
    name = "mobile_srcs",
    srcs = glob(["*.h"]),
    visibility = ["//visibility:public"],
)

alias(
    name = "android_srcs",
    actual = ":mobile_srcs",
    visibility = ["//visibility:public"],
)
