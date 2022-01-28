load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")

package(
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
    compatible_with = get_compatible_with_cloud(),
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:ArithmeticOpsTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "ptq_fallback_to_flex_inc_gen",
    compatible_with = get_compatible_with_cloud(),
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
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:export_tf_dialect_op",
        "@com_google_absl//absl/base:core_headers",
        "@flatbuffers",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)
