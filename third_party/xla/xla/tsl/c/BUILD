# Description:
# C API for TensorFlow, for use by client language bindings.

load("@local_tsl//tsl/platform:build_config.bzl", "tsl_cc_test")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility", "tsl_copts", "tsl_gpu_library")

# buildifier: disable=same-origin-load
load("//xla/tsl:tsl.default.bzl", "filegroup")

package(
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
    visibility = internal_visibility(["//tensorflow:__subpackages__"]),
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
    visibility = internal_visibility([
        "//tensorflow/c:__subpackages__",
    ]),
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
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "tsl_status",
    srcs = ["tsl_status.cc"],
    hdrs = ["tsl_status.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tsl_status_internal",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
)

tsl_cc_test(
    name = "tsl_status_test",
    srcs = ["tsl_status_test.cc"],
    deps = [
        ":tsl_status",
        ":tsl_status_internal",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
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
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
)

filegroup(
    name = "tsl_status_internal_headers",
    srcs = ["tsl_status_internal.h"],
    visibility = internal_visibility([
        "//tensorflow/c:__subpackages__",
    ]),
)
