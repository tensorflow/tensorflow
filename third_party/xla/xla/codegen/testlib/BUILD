load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
    name = "kernel_runner",
    testonly = 1,
    srcs = ["kernel_runner.cc"],
    hdrs = ["kernel_runner.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)
