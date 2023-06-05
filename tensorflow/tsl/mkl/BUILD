load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "//tensorflow/tsl:tsl.bzl",
    "clean_dep",
)

licenses(["notice"])  # 3-Clause BSD

config_setting(
    name = "build_with_mkl",
    define_values = {
        "build_with_mkl": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "build_with_mkl_lnx_x64",
    define_values = {
        "build_with_mkl": "true",
    },
    values = {
        "cpu": "k8",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "build_with_mkl_lnx_openmp",
    constraint_values = [
        "@platforms//os:linux",
    ],
    define_values = {
        "build_with_mkl": "true",
        "build_with_openmp": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "build_with_mkl_windows_openmp",
    constraint_values = [
        "@platforms//os:windows",
    ],
    define_values = {
        "build_with_mkl": "true",
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

config_setting(
    name = "enable_mkl",
    define_values = {
        "enable_mkl": "true",
        "build_with_mkl": "true",
    },
    visibility = ["//visibility:public"],
)

filegroup(
    name = "LICENSE",
    srcs = [
        "MKL_LICENSE",
        "@llvm_openmp//:LICENSE.txt",
    ],
    visibility = ["//visibility:public"],
)

# TODO(Intel-tf) Remove the following 3 calls to cc_library and replace all uses
# of mkl_libs_* with @llvm_openmp//:libiomp5.* directly.

cc_library(
    name = "mkl_libs_linux",
    srcs = [
        "@llvm_openmp//:libiomp5.so",
    ],
    hdrs = ["@llvm_openmp//:config_omp"],
    target_compatible_with = select({
        "//tensorflow/tsl/mkl:build_with_mkl": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
)

# MacOS build configuration is provided for completness, it has not been tested
cc_library(
    name = "mkl_libs_darwin",
    srcs = [
        "@llvm_openmp//:libiomp5.dylib",
    ],
    hdrs = ["@llvm_openmp//:config_omp"],
    target_compatible_with = select({
        "//tensorflow/tsl/mkl:build_with_mkl": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl_libs_windows",
    srcs = [
        "@llvm_openmp//:libiomp5md.dll",
    ],
    hdrs = ["@llvm_openmp//:config_omp"],
    target_compatible_with = select({
        "//tensorflow/tsl/mkl:build_with_mkl": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "intel_binary_blob",
    target_compatible_with = select({
        "//tensorflow/tsl/mkl:build_with_mkl": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        clean_dep("//tensorflow/tsl:linux_x86_64"): [
            ":mkl_libs_linux",
        ],
        clean_dep("//tensorflow/tsl:macos"): [
            ":mkl_libs_darwin",
        ],
        clean_dep("//tensorflow/tsl:windows"): [
            ":mkl_libs_windows",
        ],
        "//conditions:default": [],
    }),
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)
