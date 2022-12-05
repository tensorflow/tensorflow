load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_cloud")
load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "error_util",
    srcs = ["error_util.cc"],
    hdrs = ["error_util.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_test(
    name = "error_util_test",
    srcs = ["error_util_test.cc"],
    deps = [
        ":error_util",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
