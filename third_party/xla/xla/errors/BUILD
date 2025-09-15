load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "error_codes",
    hdrs = ["error_codes.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_cc_test(
    name = "error_codes_test",
    srcs = ["error_codes_test.cc"],
    deps = [
        ":error_codes",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
