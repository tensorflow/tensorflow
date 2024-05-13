# Groups the implementations of the HLO to TF operation conversions.
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "util",
    srcs = [
        "util.cc",
    ],
    hdrs = [
        "util.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/algorithm:container",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "scatter",
    srcs = [
        "scatter.cc",
    ],
    hdrs = [
        "scatter.h",
    ],
    deps = [
        ":util",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "reduce",
    srcs = [
        "reduce.cc",
    ],
    hdrs = [
        "reduce.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite/stablehlo:hlo_matchers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "dot_general",
    srcs = [
        "dot_general.cc",
    ],
    hdrs = ["dot_general.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "custom_call",
    srcs = [
        "custom_call.cc",
    ],
    hdrs = [
        "custom_call.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)
