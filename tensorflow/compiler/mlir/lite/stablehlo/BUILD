load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "drop_savedmodel_semantics",
    srcs = [
        "transforms/drop_savedmodel_semantics.cc",
    ],
    hdrs = [
        "transforms/drop_savedmodel_semantics.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":stablehlo_util",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stablehlo_tfl",
    srcs = [
        "transforms/stablehlo_tfl_pass.cc",
    ],
    hdrs = [
        "transforms/stablehlo_tfl_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@flatbuffers",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stablehlo_util",
    srcs = [
        "transforms/stablehlo_util.cc",
    ],
    hdrs = [
        "transforms/stablehlo_util.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rename_entrypoint_to_main",
    srcs = [
        "transforms/rename_entrypoint_to_main.cc",
    ],
    hdrs = [
        "transforms/rename_entrypoint_to_main.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":stablehlo_util",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "smuggle_disallowed_ops",
    srcs = [
        "transforms/smuggle_disallowed_ops.cc",
    ],
    hdrs = [
        "transforms/smuggle_disallowed_ops.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":stablehlo_util",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_stablehlo",
    srcs = [
        "transforms/tf_stablehlo_pass.cc",
    ],
    hdrs = [
        "transforms/tf_stablehlo_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":stablehlo_util",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/mlir_hlo:mhlo_passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:register",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tfl_stablehlo",
    srcs = [
        "transforms/tfl_stablehlo_pass.cc",
    ],
    hdrs = [
        "transforms/tfl_stablehlo_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "@flatbuffers",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "transforms",
    srcs = [
        "transforms/transforms.cc",
    ],
    hdrs = [
        "transforms/transforms.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":drop_savedmodel_semantics",
        ":fold_broadcast_pass",
        ":fuse_convolution_pass",
        ":optimize",
        ":rename_entrypoint_to_main",
        ":smuggle_disallowed_ops",
        ":tf_stablehlo",
        ":unfuse_batch_norm_pass",
        "//tensorflow/compiler/mlir/quantization/tensorflow:passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "//tensorflow/compiler/mlir/xla:tf_xla_passes",
        "//tensorflow/compiler/xla/mlir_hlo:mhlo_passes",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "check_accepted_ops_pass",
    srcs = [
        "transforms/check_accepted_ops_pass.cc",
    ],
    hdrs = [
        "transforms/check_accepted_ops_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":stablehlo_util",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "op_stat_pass",
    srcs = [
        "transforms/op_stat_pass.cc",
    ],
    hdrs = [
        "transforms/op_stat_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":stablehlo_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "unfuse_batch_norm_pass",
    srcs = [
        "transforms/unfuse_batch_norm_pass.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//tensorflow/compiler/xla/mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "fold_broadcast_pass",
    srcs = [
        "transforms/fold_broadcast_pass.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//tensorflow/compiler/xla/mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "fuse_convolution_pass",
    srcs = [
        "transforms/fuse_convolution_pass.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:validators",
        "//tensorflow/compiler/xla/mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "optimize",
    srcs = [
        "transforms/optimize.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//tensorflow/compiler/xla/mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "odml_to_stablehlo",
    srcs = [
        "odml_to_stablehlo.cc",
    ],
    copts = ["-O3"],
    deps = [
        ":check_accepted_ops_pass",
        ":op_stat_pass",
        ":stablehlo_tfl",
        ":transforms",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir:passes",
        "//tensorflow/compiler/mlir/lite:flatbuffer_export",
        "//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir/lite/stablehlo/serializer:flatbuffer_export",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantize_preprocess",
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_quant_ops",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_graph_optimization_pass",
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "//tensorflow/compiler/mlir/xla:legalize_tf",
        "//tensorflow/compiler/xla/mlir/framework/transforms:passes",
        "//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:register",
    ],
)

tf_cc_binary(
    name = "odml-to-stablehlo-opt",
    testonly = True,
    tags = ["hostonly"],
    deps = [
        ":fold_broadcast_pass",
        ":fuse_convolution_pass",
        ":optimize",
        ":stablehlo_tfl",
        ":tf_stablehlo",
        ":tfl_stablehlo",
        ":unfuse_batch_norm_pass",
        "//tensorflow/compiler/mlir:passes",
        "//tensorflow/compiler/mlir:tf_mlir_opt_main",
    ],
)

exports_files([
    "run_lit.sh",
])
