# Description:
#    TF2XLA Bridge and related components.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
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

cc_library(
    name = "compile_mlir_util_no_tf_dialect_passes",
    srcs = ["api/v0/compile_mlir_util.cc"],
    hdrs = ["api/v0/compile_mlir_util.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:bridge_logger",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:export_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:serialize_mlir_module_utils",
        "//tensorflow/compiler/mlir/tensorflow:shape_inference_pass",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tensorflow:translate_utils",
        "//tensorflow/compiler/mlir/tf2xla:mlir_bridge_rollout_policy",
        "//tensorflow/compiler/mlir/xla:tf_xla_passes",
        "//tensorflow/compiler/mlir/xla:xla_legalize_targets",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf_with_tf2xla",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/compiler/tf2xla:xla_argument",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/mlir/framework/transforms:passes",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/mlir_hlo:mhlo_passes",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:layout_util",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:type_to_shape",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/platform:error_payloads",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:register",
    ],
)

alias(
    name = "compile_mlir_util",
    actual = ":compile_mlir_util_no_tf_dialect_passes",
)
