# Tensorflow default + linux implementations of tensorflow/core/platform libraries.
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_visibility = [
        "//tensorflow/core/platform:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

bzl_library(
    name = "cuda_build_defs_bzl",
    srcs = ["cuda_build_defs.bzl"],
    visibility = ["//tensorflow:__subpackages__"],
)

package_group(
    name = "core_and_platform_packages",
    packages = [
        "//tensorflow/core",
        "//tensorflow/core/platform",
    ],
)

exports_files(
    srcs = glob(
        ["*"],
        exclude = [
            "integral_types.h",
            "logging.h",
            "test.cc",
            "test_benchmark.cc",
        ],
    ),
    visibility = ["//tensorflow/core/platform:__pkg__"],
)

exports_files(
    srcs = [
        "integral_types.h",
        "logging.h",
        "test.cc",
        "test_benchmark.cc",
    ],
    visibility = [":core_and_platform_packages"],
)
