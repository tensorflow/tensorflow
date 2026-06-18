load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "error_codes",
    hdrs = ["error_codes.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform",
    ] + if_google([
        "@com_google_absl//absl/types:source_location",
    ]),
)

xla_cc_test(
    name = "error_codes_test",
    srcs = ["error_codes_test.cc"],
    deps = [
        ":error_codes",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform",
    ],
)

cc_library(
    name = "debug_me_context_util",
    srcs = ["debug_me_context_util.cc"],
    hdrs = ["debug_me_context_util.h"],
    deps = [
        "//xla/tsl/platform:debug_me_context",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@tsl//tsl/platform",
    ],
)

xla_cc_test(
    name = "debug_me_context_util_test",
    srcs = ["debug_me_context_util_test.cc"],
    deps = [
        ":debug_me_context_util",
        "//xla/tsl/platform:debug_me_context",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform",
    ],
)

cc_library(
    name = "fatal_error_sink",
    srcs = ["fatal_error_sink.cc"],
    hdrs = ["fatal_error_sink.h"],
    deps = [
        ":debug_me_context_util",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/log:log_entry",
        "@com_google_absl//absl/log:log_sink",
        "@com_google_absl//absl/log:log_sink_registry",
    ],
)

xla_cc_test(
    name = "fatal_error_sink_test",
    srcs = ["fatal_error_sink_test.cc"],
    deps = [
        ":debug_me_context_util",
        ":fatal_error_sink",
        "//xla/tsl/platform:debug_me_context",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log_sink_registry",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "check",
    hdrs = ["check.h"],
    deps = ["//xla/error/internal:check_impl"],
)

xla_cc_test(
    name = "check_test",
    srcs = ["check_test.cc"],
    deps = [
        ":check",
        ":debug_me_context_util",
        "//xla/tsl/platform:debug_me_context",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fatal_error_sink_registration",
    srcs = ["fatal_error_sink_registration.cc"],
    deps = [":fatal_error_sink"],
    alwayslink = True,
)
