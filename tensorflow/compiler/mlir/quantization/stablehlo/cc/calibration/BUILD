load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:__subpackages__",
        "//tensorflow/compiler/mlir/quantization/tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "min_max_value",
    srcs = [],
    hdrs = ["min_max_value.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [],
)

cc_library(
    name = "statistics",
    srcs = ["statistics.cc"],
    hdrs = ["statistics.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:graph_def",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibration_statistics_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibrator_singleton",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "assign_ids",
    srcs = ["assign_ids.cc"],
    hdrs = ["assign_ids.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:graph_def",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibrator_singleton",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "assign_ids_test",
    srcs = ["assign_ids_test.cc"],
    deps = [
        ":assign_ids",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibrator_singleton_impl",
        "//tensorflow/core:protos_all_cc",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:protobuf",
    ],
)
