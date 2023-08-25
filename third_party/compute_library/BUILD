load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

exports_files(["LICENSE"])

config_setting(
    name = "build_with_acl",
    define_values = {
        "build_with_acl": "true",
    },
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)
