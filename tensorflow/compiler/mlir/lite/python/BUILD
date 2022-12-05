load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

licenses(["notice"])

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
)

package_group(
    name = "friends",
    packages = [
        "//learning/brain/mobile/lite/tools/analyzer/...",
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
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/toco:types_proto_cc",
        "//tensorflow/lite/tools/optimize:reduced_precision_support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
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
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/toco:types_proto_cc",
        "//tensorflow/lite/tools/optimize:reduced_precision_support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
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
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/toco:types_proto_cc",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "jax_to_tfl_flatbuffer",
    srcs = ["jax_to_tfl_flatbuffer.cc"],
    hdrs = [
        "jax_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":tf_tfl_flatbuffer_helpers",
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tf_tfl_passes",
        "//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/toco:types_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

# Smaller version of flatbuffer_translate which only converts flatbuffer to MLIR.
cc_library(
    name = "flatbuffer_to_mlir",
    srcs = [
        "flatbuffer_to_mlir.cc",
    ],
    hdrs = [
        "flatbuffer_to_mlir.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:flatbuffer_import",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)
