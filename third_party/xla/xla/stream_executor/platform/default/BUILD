load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "tsl_copts")

licenses(["notice"])

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//xla/stream_executor:__subpackages__",
    ],
)

cc_library(
    name = "platform",
    textual_hdrs = ["initialize.h"],
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
    deps = [
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:dso_loader",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)
