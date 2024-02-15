load("@local_tsl//tsl:tsl.default.bzl", "filegroup")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

filegroup(
    name = "header_and_client",
    srcs = glob([
        "c_api*",
        "libtpu*",
    ]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libtpu",
    hdrs = ["libtpu.h"],
)
