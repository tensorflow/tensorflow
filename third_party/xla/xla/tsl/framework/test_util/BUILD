# Description:
#   Test util libraries.

load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "mock_serving_device_selector",
    testonly = 1,
    hdrs = ["mock_serving_device_selector.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tsl/framework:serving_device_selector",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)
