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
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:io",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration:min_max_value",
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_quant_ops",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibration_statistics_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:path",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "component",
    srcs = ["component.cc"],
    hdrs = ["component.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":representative_dataset",
        ":statistics",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo:tf_passes",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:component",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:debugger",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:io",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:saved_model_export",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:types",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibration_statistics_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:run_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "representative_dataset",
    srcs = ["representative_dataset.cc"],
    hdrs = ["representative_dataset.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "representative_dataset_test",
    srcs = ["representative_dataset_test.cc"],
    deps = [
        ":representative_dataset",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "calibration_parameters",
    srcs = [],
    hdrs = ["calibration_parameters.h"],
    compatible_with = get_compatible_with_portable(),
    deps = ["//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc"],
)

tf_cc_test(
    name = "calibration_parameters_test",
    srcs = ["calibration_parameters_test.cc"],
    deps = [
        ":calibration_parameters",
        "@com_google_googletest//:gtest_main",
    ],
)
