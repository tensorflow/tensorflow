load(
    "//xla/tsl/platform:build_config.bzl",
    "tsl_cc_test",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "temporary_directory",
    testonly = True,
    srcs = ["temporary_directory.cc"],
    hdrs = ["temporary_directory.h"],
    deps = [
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_for_library",
        "@tsl//tsl/platform:path",
    ],
)

tsl_cc_test(
    name = "temporary_directory_test",
    srcs = ["temporary_directory_test.cc"],
    deps = [
        ":temporary_directory",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:file_statistics",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
