# Description:
# Platform-specific build configurations.

load("//tsl:tsl.bzl", "set_external_visibility", "tsl_copts")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(
    ["LICENSE"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensorflow_platform_specific",
    copts = tsl_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "_empty_lib",
    visibility = ["//visibility:public"],
)

# Minimal lib so that tools used for mobile compilation
# don't have to depend on platformlib.
cc_library(
    name = "proto_parsing",
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/protobuf:protos_all_cc",
    ],
)

cc_library(
    name = "test_main",
    testonly = 1,
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [],
)
