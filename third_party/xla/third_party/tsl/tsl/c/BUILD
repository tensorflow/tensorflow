# Description:
# C API for TensorFlow, for use by client language bindings.

load("//tsl/platform:build_config.bzl", "tsl_cc_test")
load("//tsl/platform:rules_cc.bzl", "cc_library")
load("//tsl:tsl.bzl", "tsl_copts", "tsl_gpu_library")

# buildifier: disable=same-origin-load
load("//tsl:tsl.default.bzl", "filegroup")

package(
    default_visibility = ["//visibility:public"],
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

# -----------------------------------------------------------------------------
# Public targets

filegroup(
    name = "headers",
    srcs = [
        "tsl_status.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = [
            "*test*",
        ],
    ),
    visibility = ["//visibility:public"],
)

tsl_gpu_library(
    name = "c_api",
    hdrs = [
        "tsl_status.h",
    ],
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":tsl_status_internal",
    ],
)

tsl_gpu_library(
    name = "tsl_status_internal",
    hdrs = [
        "tsl_status.h",
        "tsl_status_internal.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/platform:status",
    ],
)

cc_library(
    name = "tsl_status",
    srcs = ["tsl_status.cc"],
    hdrs = ["tsl_status.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tsl_status_internal",
        "//tsl/platform:errors",
        "//tsl/platform:status",
    ],
)

tsl_cc_test(
    name = "tsl_status_test",
    srcs = ["tsl_status_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":tsl_status",
        ":tsl_status_internal",
        "//tsl/platform:errors",
        "//tsl/platform:status",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

cc_library(
    name = "tsl_status_headers",
    hdrs = ["tsl_status.h"],
    visibility = ["//visibility:public"],
)

tsl_gpu_library(
    name = "tsl_status_helper",
    srcs = ["tsl_status_helper.cc"],
    hdrs = ["tsl_status_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tsl_status",
        ":tsl_status_internal",
        "//tsl/platform:errors",
        "//tsl/platform:status",
    ],
)

filegroup(
    name = "tsl_status_internal_headers",
    srcs = ["tsl_status_internal.h"],
    visibility = ["//visibility:public"],
)
