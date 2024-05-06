load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//learning/brain/mobile/model_optimization/...",
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/mlir/lite/...",
    ],
)

td_library(
    name = "ptq_td_files",
    srcs = [
        "fallback_to_flex_patterns.td",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

gentbl_cc_library(
    name = "ptq_fallback_to_flex_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "fallback_to_flex_patterns.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "fallback_to_flex_patterns.td",
    deps = [":ptq_td_files"],
)

cc_library(
    name = "tf_quantization_passes",
    srcs = [
        "fallback_to_flex_ops.cc",
        "fallback_to_flex_patterns.inc",
        "tf_to_quant.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":ptq_fallback_to_flex_inc_gen",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:export_tf_dialect_op",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)
