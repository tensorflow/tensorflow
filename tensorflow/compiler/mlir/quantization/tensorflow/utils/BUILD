load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:internal_visibility_allowlist_package",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "fake_quant_utils",
    srcs = ["fake_quant_utils.cc"],
    hdrs = ["fake_quant_utils.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "lift_as_function_call_utils_td_files",
    srcs = [
        "lift_as_function_call_utils.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "lift_as_function_call_utils",
    srcs = ["lift_as_function_call_utils.cc"],
    hdrs = ["lift_as_function_call_utils.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/quantization/tensorflow:pass_utils",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_to_uniform_attribute_utils",
    srcs = ["tf_to_uniform_attribute_utils.cc"],
    hdrs = ["tf_to_uniform_attribute_utils.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/quantization/tensorflow:pass_utils",
        "//tensorflow/compiler/mlir/quantization/tensorflow:uniform_op_quant_spec",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_to_xla_attribute_utils",
    srcs = ["tf_to_xla_attribute_utils.cc"],
    hdrs = ["tf_to_xla_attribute_utils.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:pass_utils",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/lite/kernels:padding",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "tf_to_xla_attribute_utils_test",
    srcs = ["tf_to_xla_attribute_utils_test.cc"],
    deps = [
        ":tf_to_xla_attribute_utils",
        "//tensorflow/compiler/mlir/quantization/tensorflow:pass_utils",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
