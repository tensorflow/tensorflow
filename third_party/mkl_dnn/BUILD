load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

config_setting(
    name = "build_with_mkl_dnn_only",
    define_values = {
        "build_with_mkl": "true",
        "build_with_mkl_dnn_only": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "build_with_mkl_dnn_v1_only",
    define_values = {
        "build_with_mkl": "true",
        "build_with_mkl_dnn_v1_only": "true",
    },
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)
