load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

config_setting(
    name = "build_with_mkl_opensource",
    define_values = {
        "build_with_mkl": "true",
        "build_with_mkl_opensource": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "build_with_mkldnn_openmp",
    define_values = {
        "build_with_mkl": "true",
        "build_with_openmp": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "build_with_onednn_v3",
    define_values = {
        "build_with_mkl": "true",
        "build_with_onednn_v3": "true",
    },
    visibility = ["//visibility:public"],
)

# The following config is needed since oneDNN v2.x and v3.x are API incompatible.
config_setting(
    name = "build_with_onednn_v2",
    define_values = {
        # We are not defining 'build_with_mkl' since this config can be invoked
        # on x86_64 platforms without --config=mkl
        "build_with_onednn_v2": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "build_with_mkl_aarch64_openmp",
    define_values = {
        "build_with_mkl_aarch64": "true",
        "build_with_openmp": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "build_with_mkl_aarch64",
    define_values = {
        "build_with_mkl_aarch64": "true",
    },
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)
