# NVIDIA TensorRT
# A high-performance deep learning inference optimizer and runtime.

licenses(["notice"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

cc_library(
    name = "tensorrt_headers",
    hdrs = [
        "tensorrt/include/tensorrt_config.h",
        ":tensorrt_include"
    ],
    include_prefix = "third_party/tensorrt",
    strip_include_prefix = "tensorrt/include",
)

cc_library(
    name = "tensorrt",
    srcs = [":tensorrt_lib"],
    copts = cuda_default_copts(),
    data = [":tensorrt_lib"],
    linkstatic = 1,
    deps = [
        ":tensorrt_headers",
        "@local_config_cuda//cuda",
    ],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    deps = [
        "@bazel_skylib//lib:selects",
    ],
)

%{copy_rules}
