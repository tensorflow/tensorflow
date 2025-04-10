# Description:
#   Tensorflow / MLIR utils.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "name_utils",
    srcs = ["name_utils.cc"],
    hdrs = ["name_utils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "string_container_utils",
    hdrs = ["string_container_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "array_container_utils",
    hdrs = ["array_container_utils.h"],
    deps = [
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "saved_model_converter_utils",
    srcs = ["saved_model_converter_utils.cc"],
    hdrs = ["saved_model_converter_utils.h"],
    visibility = [
        "//tensorflow/cc/experimental/tfa:__subpackages__",
    ],
    deps = [
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:mlir_roundtrip_flags",
        "//tensorflow/core/framework:op",
        "//tensorflow/core/framework:op_def_builder",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "validators",
    srcs = [
        "validators.cc",
    ],
    hdrs = [
        "validators.h",
    ],
    deps = [
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
