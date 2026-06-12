load("//xla:xla.default.bzl", "xla_cc_binary", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "compare_literals",
    srcs = ["compare_literals.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":compare_literals_impl",
        "//xla:error_spec",
        "//xla:xla_data_proto_cc",
        "//xla/tools:run_hlo_module_proto_cc",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "compare_literals_impl",
    srcs = ["compare_literals_impl.cc"],
    hdrs = ["compare_literals_impl.h"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_comparison",
        "//xla:xla_data_proto_cc",
        "//xla/tools:run_hlo_module_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "compare_literals_impl_test",
    srcs = ["compare_literals_impl_test.cc"],
    deps = [
        ":compare_literals_impl",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
