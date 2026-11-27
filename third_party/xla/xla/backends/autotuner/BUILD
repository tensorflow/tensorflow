load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "hlo_extractor",
    hdrs = ["hlo_extractor.h"],
    tags = if_google(["avoid_dep"]),
    deps = ["//xla/backends/core/autotuner:hlo_extractor"],
)
