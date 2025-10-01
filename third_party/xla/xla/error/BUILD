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
        ":debug_me_context_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_cc_test(
    name = "error_codes_test",
    srcs = ["error_codes_test.cc"],
    deps = [
        ":debug_me_context_util",
        ":error_codes",
        "//xla/tsl/platform:debug_me_context",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "debug_me_context_util",
    srcs = ["debug_me_context_util.cc"],
    hdrs = ["debug_me_context_util.h"],
    deps = [
        "//xla/tsl/platform:debug_me_context",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ],
)

xla_cc_test(
    name = "debug_me_context_util_test",
    srcs = ["debug_me_context_util_test.cc"],
    deps = [
        ":debug_me_context_util",
        "//xla/tsl/platform:debug_me_context",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
