load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

config_setting(
    name = "build_with_mkl_opensource",
    define_values = {
        "build_with_mkl": "true",
        "build_with_mkl_opensource": "true",
    },
)

config_setting(
    name = "build_with_mkldnn_openmp",
    define_values = {
        "build_with_mkl": "true",
        "build_with_openmp": "true",
    },
)

config_setting(
    name = "build_with_mkl_aarch64_openmp",
    define_values = {
        "build_with_mkl_aarch64": "true",
        "build_with_openmp": "true",
    },
)

config_setting(
    name = "build_with_mkl_aarch64",
    define_values = {
        "build_with_mkl_aarch64": "true",
    },
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)
