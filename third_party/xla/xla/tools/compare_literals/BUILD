# Tools for comparing XLA output literals.

load(
    "//xla:xla.default.bzl",
    "xla_cc_binary",
    "xla_cc_test",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "compare_literals_lib",
    srcs = [
        "compare_literal_math.cc",
        "compare_literal_report.cc",
        "compare_literals.cc",
    ],
    hdrs = [
        "compare_literals.h",
        "element_comparator.h",
    ],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_binary(
    name = "compare_literals",
    srcs = ["compare_literals_main.cc"],
    deps = [
        ":compare_literals_lib",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_cc_test(
    name = "compare_literals_test",
    srcs = ["compare_literals_test.cc"],
    deps = [
        ":compare_literals_lib",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
    ],
)
