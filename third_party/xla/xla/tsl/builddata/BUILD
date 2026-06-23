load("//xla/tsl:tsl.bzl", "if_oss", "tsl_copts")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_platform_deps",
    "tsl_cc_test",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "builddata_headers",
    hdrs = ["builddata.h"],
    visibility = [
        "//xla/tsl:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "builddata_globals_hdrs",
    hdrs = [
        "builddata_globals.h",
        "utils.h",
    ],
    visibility = [
        "//xla/tsl:__subpackages__",
    ],
)

cc_library(
    name = "builddata",
    hdrs = [
        "builddata.h",
    ],
    copts = tsl_copts(),
    linkstamp = if_oss(
        "builddata_globals.cc",
        google_value = None,
    ),
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ] + tf_platform_deps("builddata"),
)

tsl_cc_test(
    name = "test_stamping",
    srcs = [
        "test_stamping.cc",
        "utils.h",
    ],
    deps = [
        ":builddata",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
