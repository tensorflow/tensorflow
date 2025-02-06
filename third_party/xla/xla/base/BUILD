# Common utilities for implementing XLA.

load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "fixed_options_flag",
    srcs = ["fixed_options_flag.cc"],
    hdrs = ["fixed_options_flag.h"],
    compatible_with = ["//buildenv/target:all"],
    deps = [
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "fixed_options_flag_test",
    srcs = ["fixed_options_flag_test.cc"],
    deps = [
        ":fixed_options_flag",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
