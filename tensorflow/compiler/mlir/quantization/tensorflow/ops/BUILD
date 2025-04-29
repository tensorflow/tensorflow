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
    name = "tf_op_quant_spec",
    srcs = [
        "tf_op_quant_spec.cc",
    ],
    hdrs = ["tf_op_quant_spec.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "tf_op_quant_spec_test",
    srcs = ["tf_op_quant_spec_test.cc"],
    deps = [
        ":tf_op_quant_spec",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "temp_tf_op_quant_spec",
    srcs = [
        "temp_tf_op_quant_spec.cc",
    ],
    hdrs = ["temp_tf_op_quant_spec.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/common/tf_quantization_lib",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "temp_tf_op_quant_spec_test",
    srcs = ["temp_tf_op_quant_spec_test.cc"],
    deps = [
        ":temp_tf_op_quant_spec",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tf_quantize_op",
    srcs = [
        "tf_quantize_op.cc",
    ],
    hdrs = ["tf_quantize_op.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/utils:tf_quantize_op_utils",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "tf_quantize_op_test",
    srcs = ["tf_quantize_op_test.cc"],
    deps = [
        ":tf_quantize_op",
        "//tensorflow/compiler/mlir/quantization/common:attrs_and_constraints",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tf_uniform_op_quant_spec",
    srcs = [
        "tf_uniform_op_quant_spec.cc",
    ],
    hdrs = ["tf_uniform_op_quant_spec.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/common/tf_quantization_lib",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "uniform_op_quant_spec",
    srcs = [
        "uniform_op_quant_spec.cc",
    ],
    hdrs = ["uniform_op_quant_spec.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
