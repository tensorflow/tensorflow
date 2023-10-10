load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_tsl//tsl:tsl.bzl", "tsl_copts")

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "platform",
    textual_hdrs = ["initialize.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "dso_loader",
    hdrs = ["dso_loader.h"],
    compatible_with = [],
    copts = tsl_copts(),
    tags = [
        "manual",
        "nobuilder",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:dso_loader",
    ],
)
