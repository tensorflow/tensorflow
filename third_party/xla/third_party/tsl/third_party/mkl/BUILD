licenses(["notice"])  # 3-Clause BSD

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(default_visibility = ["//visibility:public"])

alias(
    name = "build_with_mkl",
    actual = "//tsl/mkl:build_with_mkl",
    visibility = ["//visibility:public"],
)

alias(
    name = "build_with_mkl_lnx_x64",
    actual = "//tsl/mkl:build_with_mkl_lnx_x64",
    visibility = ["//visibility:public"],
)

alias(
    name = "build_with_mkl_lnx_openmp",
    actual = "//tsl/mkl:build_with_mkl_lnx_openmp",
    visibility = ["//visibility:public"],
)

alias(
    name = "build_with_mkl_windows_openmp",
    actual = "//tsl/mkl:build_with_mkl_windows_openmp",
    visibility = ["//visibility:public"],
)

alias(
    name = "build_with_mkl_aarch64",
    actual = "//tsl/mkl:build_with_mkl_aarch64",
    visibility = ["//visibility:public"],
)

alias(
    name = "enable_mkl",
    actual = "//tsl/mkl:enable_mkl",
    visibility = ["//visibility:public"],
)

alias(
    name = "intel_binary_blob",
    actual = "//tsl/mkl:intel_binary_blob",
    visibility = ["//visibility:public"],
)

alias(
    name = "LICENSE",
    actual = "//tsl/mkl:LICENSE",
    visibility = ["//visibility:public"],
)

alias(
    name = "mkl_libs_linux",
    actual = "//tsl/mkl:mkl_libs_linux",
    visibility = ["//visibility:public"],
)

alias(
    name = "mkl_libs_darwin",
    actual = "//tsl/mkl:mkl_libs_darwin",
    visibility = ["//visibility:public"],
)

alias(
    name = "mkl_libs_windows",
    actual = "//tsl/mkl:mkl_libs_windows",
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)
