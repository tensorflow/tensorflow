load("@local_tsl//tsl:tsl.bzl", "internal_visibility")
load("@local_tsl//tsl:tsl.default.bzl", "get_compatible_with_portable")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//third_party/golang/github_com/gomlx/gomlx:__subpackages__",
        "//xla:internal",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "error_util",
    srcs = ["error_util.cc"],
    hdrs = ["error_util.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_test(
    name = "error_util_test",
    srcs = ["error_util_test.cc"],
    deps = [
        ":error_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:test_main",
    ],
)
