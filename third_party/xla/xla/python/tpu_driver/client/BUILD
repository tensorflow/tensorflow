load("@local_tsl//tsl:tsl.default.bzl", "filegroup")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
)
