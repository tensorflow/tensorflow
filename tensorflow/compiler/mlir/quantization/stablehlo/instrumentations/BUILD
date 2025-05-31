load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["@stablehlo//:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "save_report",
    srcs = ["save_report.cc"],
    hdrs = ["save_report.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:tf_report",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "save_report_test",
    srcs = ["save_report_test.cc"],
    deps = [
        ":save_report",
        "//tensorflow/compiler/mlir/quantization/common:tf_test_base",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo:tf_passes",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:io",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/tsl/platform:status_matchers",
    ],
)
