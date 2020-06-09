# Description:
#   TensorFlow/TensorFlow Lite/XLA MLIR dialects and tools.

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = [
        "//tensorflow/compiler/tf2xla:__subpackages__",
        "@llvm-project//mlir:friends",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "subpackages",
    packages = ["//tensorflow/compiler/mlir/..."],
)

exports_files(glob(["g3doc/*.md"]))

# To reference all tablegen files here when checking for updates to them.
filegroup(
    name = "td_files",
    srcs = glob(["**/*.td"]),
)

cc_library(
    name = "op_or_arg_name_mapper",
    srcs = ["op_or_arg_name_mapper.cc"],
    hdrs = ["op_or_arg_name_mapper.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_mlir_opt_main",
    srcs = ["tf_mlir_opt_main.cc"],
    deps = [
        ":init_mlir",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:logging",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "passes",
    visibility = [
        ":__subpackages__",
        "//tensorflow/python:__subpackages__",
    ],
    deps = [
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:QuantOps",
        # Link jit lib to link JIT devices required to run
        # xla-legalize-tf-with-tf2xla pass.
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_dialect_registration",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_legalize_tf",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_optimize",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_quantize",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_passes",
        "//tensorflow/compiler/mlir/lite/quantization/tensorflow:tf_to_quant",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_dialect_registration",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_test_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_legalize_hlo",
        "//tensorflow/compiler/mlir/tfjs:tensorflow_js_passes",
    ],
)

cc_library(
    name = "init_mlir",
    srcs = ["init_mlir.cc"],
    hdrs = ["init_mlir.h"],
    deps = [
        "//tensorflow/core:lib",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "mlir_graph_optimization_pass",
    srcs = ["mlir_graph_optimization_pass.cc"],
    hdrs = ["mlir_graph_optimization_pass.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:device_util",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/core:core_cpu",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_graph_optimization_pass_registration",
    srcs = [
        "mlir_graph_optimization_pass_registration.cc",
    ],
    deps = [
        ":mlir_graph_optimization_pass",
        "//tensorflow/core:core_cpu",
    ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "tf-opt",
    deps = [
        ":passes",
        ":tf_mlir_opt_main",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_dialect_registration",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_pass_registration",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_dialect_registration",
        "//tensorflow/compiler/mlir/tensorflow:tf_graph_optimization_pass",
        "//tensorflow/compiler/mlir/tfjs:tensorflow_js_dialect_registration",
        "//tensorflow/compiler/mlir/xla:all_xla_passes_for_testing",
    ],
)

tf_cc_binary(
    name = "tf-mlir-translate",
    srcs = ["tf_mlir_translate_main.cc"],
    deps = [
        ":init_mlir",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_dialect_registration",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_dialect_registration",
        "//tensorflow/compiler/mlir/tensorflow:translate_cl_options",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/compiler/mlir/tensorflow:translate_registration",
        "//tensorflow/compiler/mlir/tensorflow:translate_tf_dialect_op",
        "//tensorflow/compiler/mlir/xla:xla_mlir_translate",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
)

filegroup(
    name = "litfiles",
    srcs = glob(["runlit*py"]),
)
