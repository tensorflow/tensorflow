# Description:
#   TensorFlow/TensorFlow Lite/XLA MLIR dialects and tools.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cc_test",
)
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
    name = "op_or_arg_name_mapper",
    srcs = ["op_or_arg_name_mapper.cc"],
    hdrs = ["op_or_arg_name_mapper.h"],
    deps = [
        "//tensorflow/compiler/mlir/utils:name_utils",
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
        ":passes",
        ":register_common_dialects",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tf_tfl_passes",  # buildcleaner:keep
        "//tensorflow/compiler/mlir/quantization/stablehlo:bridge_passes",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_pass_registration",
        "//tensorflow/compiler/mlir/tensorflow:mlprogram_util",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_test_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_graph_optimization_pass",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_saved_model_passes",  # buildcleaner:keep
        "//tensorflow/compiler/mlir/tensorflow/transforms/host_runtime:lower_cluster_to_runtime_ops",
        "//tensorflow/compiler/mlir/tensorflow/transforms/host_runtime:runtime_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms/sparsecore:sparsecore_passes",
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "//tensorflow/compiler/mlir/tf2xla/internal/passes:clustering_passes",
        "//tensorflow/compiler/mlir/tf2xla/internal/passes:mlir_to_graph_passes",
        "//tensorflow/compiler/mlir/tf2xla/transforms:tf_xla_passes",
        "//tensorflow/compiler/mlir/tf2xla/transforms:xla_legalize_tf",
        "//tensorflow/compiler/mlir/tosa:tf_passes",
        "//tensorflow/compiler/mlir/tosa:tf_tfl_passes",
        "//tensorflow/compiler/mlir/tosa:tfl_passes",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/mlir/framework/ir:xla_framework",
        "@local_xla//xla/mlir/framework/transforms:passes",
        "@local_xla//xla/mlir_hlo:all_passes",
        "@local_xla//xla/service/cpu:hlo_xla_runtime_pipeline",
    ],
)

cc_library(
    name = "passes",
    visibility = [
        ":__subpackages__",
        "//tensorflow/python:__subpackages__",
    ],
    deps = [
        "@llvm-project//mlir:AffineDialect",
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
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
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
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:device_util",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:export_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tf2xla:mlir_bridge_rollout_policy",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ShapeDialect",
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

# This should just be a wrapper around tf_mlir_opt_main. Don't add
# direct dependencies to this binary.
tf_cc_binary(
    name = "tf-opt",
    testonly = True,
    deps = [
        ":tf_mlir_opt_main",
    ],
)

tf_cc_binary(
    name = "tf-reduce",
    testonly = True,
    srcs = ["tf_mlir_reduce_main.cc"],
    deps = [
        ":init_mlir",
        ":register_common_dialects",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_reduce_patterns_inc_gen",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirReduceLib",
        "@llvm-project//mlir:Reducer",
    ],
)

cc_library(
    name = "register_common_dialects",
    testonly = True,
    srcs = ["register_common_dialects.cc"],
    hdrs = ["register_common_dialects.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_pass_registration",
        "//tensorflow/compiler/mlir/tensorflow:mlprogram_util",
        "//tensorflow/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "@llvm-project//mlir:AllExtensions",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/mlir/framework/ir:xla_framework",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@local_xla//xla/service/cpu:hlo_xla_runtime_pipeline",
        "@stablehlo//:register",
    ],
)

tf_cc_test(
    name = "register_common_dialects_test",
    srcs = ["register_common_dialects_test.cc"],
    deps = [
        ":register_common_dialects",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_binary(
    name = "tf-mlir-translate",
    testonly = True,
    srcs = ["tf_mlir_translate_main.cc"],
    deps = [
        ":init_mlir",
        "//tensorflow/compiler/mlir/tensorflow:tf_xla_mlir_translate",
        "//tensorflow/compiler/mlir/tensorflow:translate_cl_options",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/compiler/mlir/tensorflow:translate_registration",
        "//tensorflow/compiler/mlir/tensorflow:translate_tf_dialect_op",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
        "@local_xla//xla/translate/hlo_to_mhlo:translate_registration",
        "@local_xla//xla/translate/mhlo_to_hlo:translate_registration",
    ],
)

tf_cc_test(
    name = "mlir_graph_optimization_pass_test",
    srcs = ["mlir_graph_optimization_pass_test.cc"],
    deps = [
        ":mlir_graph_optimization_pass",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:optimization_registry",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

filegroup(
    name = "litfiles",
    srcs = glob(["runlit*py"]),
    visibility = ["//tensorflow:__subpackages__"],
)

exports_files(["run_lit.sh"])
