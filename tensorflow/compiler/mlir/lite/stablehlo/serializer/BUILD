load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "flatbuffer_translator",
    srcs = [
        "flatbuffer_translator.cc",
    ],
    hdrs = ["flatbuffer_translator.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:logging",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/stablehlo/schema:schema_fbs",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "flatbuffer_export",
    srcs = [
        "flatbuffer_export.cc",
    ],
    hdrs = ["flatbuffer_export.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":flatbuffer_translator",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:logging",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//mlir:IR",
    ],
)
