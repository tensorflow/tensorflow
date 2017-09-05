# Description:
# Platform-specific build configurations.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "if_cuda")
load("//tensorflow:tensorflow.bzl", "tf_copts")
load("//tensorflow:tensorflow.bzl", "tf_cuda_library")
load("@local_config_sycl//sycl:platform.bzl", "sycl_library_path")

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
    deps = [
        "//tensorflow/stream_executor",
    ] + if_cuda(
        ["//tensorflow/stream_executor:cuda_platform"],
    ) + select({
        "@local_config_cuda//cuda:darwin": ["IOKit"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "stream_executor_cuda",
    deps = [
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor:cuda_platform",
    ] + select({
        "@local_config_cuda//cuda:darwin": ["IOKit"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "stream_executor_no_cuda",
    deps = [
        "//tensorflow/stream_executor",
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
        "//tensorflow/core:protos_cc",
        "@com_googlesource_code_re2//:re2",
        "@farmhash_archive//:farmhash",
        "@fft2d//:fft2d",
        "@highwayhash//:sip_hash",
        "@png_archive//:png",
    ],
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
        "@jpeg//:jpeg",
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
    data = [
        "@local_config_sycl//sycl:{}".format(sycl_library_path("ComputeCpp")),
    ],
    linkopts = select({
        "//conditions:default": [
            "-Wl,-rpath,../local_config_sycl/sycl/lib",
        ],
    }),
    deps = [
        "@local_config_sycl//sycl:syclrt",
    ],
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

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
