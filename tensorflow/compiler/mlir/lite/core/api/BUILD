load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
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

cc_library(
    name = "flatbuffer_conversions",
    srcs = ["flatbuffer_conversions.cc"],
    hdrs = [
        "flatbuffer_conversions.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tensorflow/compiler/mlir/lite/core/c:tflite_common",
        "//tensorflow/compiler/mlir/lite/kernels/internal:compatibility_macros",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers//:runtime_cc",
    ],
)

tf_cc_test(
    name = "flatbuffer_conversions_test",
    size = "small",
    srcs = ["flatbuffer_conversions_test.cc"],
    deps = [
        ":flatbuffer_conversions",
        "//tensorflow/compiler/mlir/lite/core/c:tflite_common",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)
