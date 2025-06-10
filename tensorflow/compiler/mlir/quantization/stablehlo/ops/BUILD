load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:internal_visibility_allowlist_package",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "stablehlo_op_quant_spec",
    srcs = [
        "stablehlo_op_quant_spec.cc",
    ],
    hdrs = ["stablehlo_op_quant_spec.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/common:lift_as_function_call",
        "//tensorflow/compiler/mlir/quantization/common:tf_attrs_and_constraints",
        "//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "//tensorflow/compiler/mlir/quantization/common/tf_quantization_lib",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:protobuf",
        "@stablehlo//:stablehlo_ops",
    ],
)

tf_cc_test(
    name = "stablehlo_op_quant_spec_test",
    srcs = ["stablehlo_op_quant_spec_test.cc"],
    deps = [
        ":stablehlo_op_quant_spec",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/compiler/mlir/quantization/common:attrs_and_constraints",
        "//tensorflow/compiler/mlir/quantization/common:func",
        "//tensorflow/compiler/mlir/quantization/common:test_base",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)
