# Description:
#   Test util libraries.

load(
    "//tsl/platform:rules_cc.bzl",
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
        "//tsl/framework:serving_device_selector",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "@com_google_absl//absl/strings:string_view",
    ],
)
