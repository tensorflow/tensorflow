load("@llvm-project//mlir:tblgen.bzl", "td_library")

package(
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:internal_visibility_allowlist_package",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "fake_quant_utils",
    srcs = ["fake_quant_utils.cc"],
    hdrs = ["fake_quant_utils.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "lift_as_function_call_utils_td_files",
    srcs = [
        "lift_as_function_call_utils.td",
    ],
    deps = [
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

cc_library(
    name = "lift_as_function_call_utils",
    srcs = ["lift_as_function_call_utils.cc"],
    hdrs = ["lift_as_function_call_utils.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "quant_spec",
    srcs = ["quant_spec.cc"],
    hdrs = ["quant_spec.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/tensorflow",
    ],
)
