load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

licenses(["notice"])

package(default_visibility = [":friends"])

package_group(
    name = "friends",
    packages = [
        "//tensorflow/lite/toco/...",
    ],
)

cc_library(
    name = "tf_tfl_flatbuffer_helpers",
    srcs = ["tf_tfl_flatbuffer_helpers.cc"],
    hdrs = ["tf_tfl_flatbuffer_helpers.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tf_tfl_passes",
        "//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir/lite/metrics:error_collector_inst",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/toco:types_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "graphdef_to_tfl_flatbuffer",
    srcs = ["graphdef_to_tfl_flatbuffer.cc"],
    hdrs = [
        "graphdef_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":tf_tfl_flatbuffer_helpers",
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tf_tfl_passes",
        "//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/toco:types_proto_cc",
        "//tensorflow/lite/tools/optimize:reduced_precision_support",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "saved_model_to_tfl_flatbuffer",
    srcs = ["saved_model_to_tfl_flatbuffer.cc"],
    hdrs = [
        "saved_model_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":tf_tfl_flatbuffer_helpers",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tf_tfl_passes",
        "//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/toco:types_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
