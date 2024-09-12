load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/lite:__subpackages__",
        "//tensorflow/lite:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(["error_reporter.h"])

filegroup(
    name = "tflite_internal_cc_3p_api_deps_src",
    srcs = [
        "error_reporter.cc",
        "error_reporter.h",
        "verifier.h",
    ],
    visibility = ["//tensorflow/lite:__pkg__"],
)

cc_library(
    name = "error_reporter",
    srcs = ["error_reporter.cc"],
    hdrs = ["error_reporter.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [],
)

exports_files(["verifier.h"])

cc_library(
    name = "verifier",
    hdrs = ["verifier.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [":error_reporter"],
)

tf_cc_test(
    name = "error_reporter_test",
    size = "small",
    srcs = ["error_reporter_test.cc"],
    deps = [
        ":error_reporter",
        "@com_google_googletest//:gtest_main",
    ],
)
