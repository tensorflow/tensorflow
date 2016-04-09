# Description:
# Platform-specific build configurations.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_copts")
load("//tensorflow:tensorflow.bzl", "tf_cuda_library")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_get_cuda_version")

cc_library(
    name = "gtest",
    testonly = 1,
    copts = tf_copts(),
    deps = [
        "//external:gtest",
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
    ],
)

cc_library(
    name = "platformlib",
    copts = tf_copts(),
    deps = [
        "@jpeg_archive//:jpeg",
        "@png_archive//:png",
        "@re2//:re2",
        "//tensorflow/core:protos_cc",
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
        "//third_party/gpus/cuda:lib64/libcudart.so" + tf_get_cuda_version(),
    ],
    linkopts = [
        "-Wl,-rpath,third_party/gpus/cuda/lib64",
    ],
    deps = [
        "//third_party/gpus/cuda:cudart",
    ],
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
