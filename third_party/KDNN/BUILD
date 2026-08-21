licenses(["notice"])

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

config_setting(
    name = "enable_kdnn",
    define_values = {
        "enable_kdnn": "true",
    },
    visibility = ["//visibility:public"],
)

# KDNN is implemented for ARM64 CPUs only. Keep the feature flag from
# selecting KDNN sources or dependencies on other CPU architectures.
config_setting(
    name = "enable_kdnn_arm64",
    define_values = {
        "enable_kdnn": "true",
    },
    constraint_values = ["@platforms//cpu:aarch64"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kdnn",
    deps = ["@kdnn//:kdnn"],
)

cc_library(
    name = "kdnn_adapter",
    hdrs = ["kdnn_adapter.h", "kdnn_threadpool.h", "kdnn_types_adapter.h", "kdnn_layout_adapter.h"],
    strip_include_prefix = "/third_party/KDNN",
    visibility = ["//visibility:public"],
    deps = ["@kdnn//:kdnn"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)
