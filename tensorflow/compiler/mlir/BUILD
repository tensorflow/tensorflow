# Description:
#   TensorFlow/TensorFlow Lite/XLA MLIR dialects and tools.

load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "subpackages",
    packages = ["//tensorflow/compiler/mlir/..."],
)

exports_files(glob(["g3doc/*.md"] + ["g3doc/_includes/tf_passes.md"]))

# To reference all tablegen files here when checking for updates to them.
filegroup(
    name = "td_files",
    srcs = glob(["**/*.td"]),
)

cc_library(
    name = "string_container_utils",
    hdrs = ["utils/string_container_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "array_container_utils",
    hdrs = ["utils/array_container_utils.h"],
    deps = [
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "name_utils",
    srcs = ["utils/name_utils.cc"],
    hdrs = ["utils/name_utils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "op_or_arg_name_mapper",
    srcs = ["op_or_arg_name_mapper.cc"],
    hdrs = ["op_or_arg_name_mapper.h"],
    deps = [
        ":name_utils",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_mlir_opt_main",
    testonly = True,
    srcs = ["tf_mlir_opt_main.cc"],
    deps = [
        ":init_mlir",
        "//tensorflow/compiler/mlir/hlo:all_passes",
        "//tensorflow/compiler/mlir/hlo:hlo_dialect_registration",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:compile_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_test_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_graph_optimization_pass",
        "//tensorflow/compiler/mlir/tensorflow:tf_legalize_hlo",
        "//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "//tensorflow/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "//tensorflow/compiler/mlir/tosa:tf_passes",
        "//tensorflow/compiler/mlir/tosa:tf_tfl_passes",
        "//tensorflow/compiler/mlir/tosa:tfl_passes",
        "//tensorflow/compiler/mlir/xla:tf_xla_passes",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "//tensorflow/compiler/mlir/xla:xla_passes",
        "//tensorflow/core:lib",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Shape",
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
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_legalize_tf",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_optimize",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_quantize",
        "//tensorflow/compiler/mlir/lite:lift_tflite_flex_ops",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_passes",
        "//tensorflow/compiler/mlir/lite/quantization/tensorflow:tf_quantization_passes",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:compile_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_legalize_hlo",
        "//tensorflow/compiler/mlir/tfjs:tensorflow_js_passes",
        "//tensorflow/compiler/mlir/tosa:tf_passes",
        "//tensorflow/compiler/mlir/tosa:tfl_passes",
    ],
)

cc_library(
    name = "init_mlir",
    srcs = ["init_mlir.cc"],
    hdrs = ["init_mlir.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "mlir_graph_optimization_pass",
    srcs = ["mlir_graph_optimization_pass.cc"],
    hdrs = ["mlir_graph_optimization_pass.h"],
    deps = [
        "//tensorflow/compiler/mlir:mlir_bridge_rollout_policy",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:device_util",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:export_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Shape",
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
    testonly = True,
    deps = [
        ":passes",
        ":tf_mlir_opt_main",
        "//tensorflow/compiler/mlir/tensorflow:bridge_pass_test_pipeline_registration",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_pass_registration",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_test_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_graph_optimization_pass",
        "//tensorflow/compiler/mlir/xla:tf_xla_passes",
        "//tensorflow/compiler/mlir/xla:xla_passes",
    ],
)

tf_cc_binary(
    name = "tf-reduce",
    testonly = True,
    srcs = ["tf_mlir_reduce_main.cc"],
    deps = [
        ":init_mlir",
        ":passes",
        "//tensorflow/compiler/mlir/hlo:all_passes",
        "//tensorflow/compiler/mlir/hlo:hlo_dialect_registration",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_reduce_patterns_inc_gen",
        "//tensorflow/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "//tensorflow/compiler/mlir/xla:tf_xla_passes",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "//tensorflow/compiler/mlir/xla:xla_passes",
        "//tensorflow/core:lib",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirReduceLib",
        "@llvm-project//mlir:Reducer",
    ],
)

tf_cc_binary(
    name = "tf-mlir-translate",
    testonly = True,
    srcs = ["tf_mlir_translate_main.cc"],
    deps = [
        ":init_mlir",
        "//tensorflow/compiler/mlir/tensorflow:export_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:tf_xla_mlir_translate",
        "//tensorflow/compiler/mlir/tensorflow:translate_cl_options",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/compiler/mlir/tensorflow:translate_registration",
        "//tensorflow/compiler/mlir/tensorflow:translate_tf_dialect_op",
        "//tensorflow/compiler/mlir/xla:translate_cl_registration",
        "//tensorflow/compiler/mlir/xla:xla_mlir_translate",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
)

cc_library(
    name = "mlir_bridge_rollout_policy",
    srcs = ["mlir_bridge_rollout_policy.cc"],
    hdrs = ["mlir_bridge_rollout_policy.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "mlir_graph_optimization_pass_test",
    srcs = ["mlir_graph_optimization_pass_test.cc"],
    deps = [
        ":mlir_graph_optimization_pass",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm-project//mlir:IR",
    ],
)

filegroup(
    name = "litfiles",
    srcs = glob(["runlit*py"]),
    visibility = ["//tensorflow:__subpackages__"],
)

exports_files(["run_lit.sh"])
