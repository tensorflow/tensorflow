licenses(["notice"])

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

config_setting(
    name = "enable_kdnn",
    define_values = {
        "enable_kdnn": "true",
    },
    visibility = ["//visibility:public"],
)

# KDNN is implemented for Linux ARM64 CPUs only. Keep the feature flag from
# selecting KDNN sources or dependencies on other platforms.
config_setting(
    name = "enable_kdnn_arm64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
    define_values = {
        "enable_kdnn": "true",
    },
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kdnn",
    deps = ["@kdnn//:kdnn"],
    target_compatible_with = select({
        ":enable_kdnn_arm64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

cc_library(
    name = "kdnn_adapter",
    hdrs = ["kdnn_adapter.h", "kdnn_threadpool.h", "kdnn_types_adapter.h", "kdnn_layout_adapter.h"],
    strip_include_prefix = "/third_party/KDNN",
    visibility = ["//visibility:public"],
    deps = ["@kdnn//:kdnn"],
    target_compatible_with = select({
        ":enable_kdnn_arm64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)
