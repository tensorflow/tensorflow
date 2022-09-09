load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
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
        ":mhlo_util",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "mhlo_tfl_legalize_patterns_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    includes = [
        "//tensorflow/compiler/xla/mlir_hlo/include/",
    ],
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_mhlo_tfl_legalize_patterns.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/mhlo_tfl_legalize_patterns.td",
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files",
        "@llvm-project//mlir:ArithmeticOpsTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "mhlo_tfl",
    srcs = [
        "transforms/mhlo_tfl_pass.cc",
    ],
    hdrs = [
        "transforms/mhlo_tfl_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:register",
        "@flatbuffers",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mhlo_util",
    srcs = [
        "transforms/mhlo_util.cc",
    ],
    hdrs = [
        "transforms/mhlo_util.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/mlir_hlo:legalize_trigonometric_to_approximation",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:register",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
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
        ":mhlo_util",
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
        ":mhlo_util",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/xla/mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_mhlo",
    srcs = [
        "transforms/tf_mhlo_pass.cc",
    ],
    hdrs = [
        "transforms/tf_mhlo_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":mhlo_util",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_mhlo_tfl",
    srcs = [
        "transforms/tf_mhlo_tfl_pass.cc",
    ],
    hdrs = [
        "transforms/tf_mhlo_tfl_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":mhlo_tfl_legalize_patterns_inc_gen",
        ":mhlo_util",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/mlir_hlo:legalize_trigonometric_to_approximation",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:register",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_poly",
    srcs = [
        "transforms/tf_poly_pass.cc",
    ],
    hdrs = [
        "transforms/tf_poly_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":mhlo_tfl_legalize_patterns_inc_gen",
        ":mhlo_util",
        ":tf_mhlo",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:register",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tfl_mhlo",
    srcs = [
        "transforms/tfl_mhlo_pass.cc",
    ],
    hdrs = [
        "transforms/tfl_mhlo_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:register",
        "@flatbuffers",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
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
        ":rename_entrypoint_to_main",
        ":smuggle_disallowed_ops",
        ":tf_mhlo",
        "//tensorflow/compiler/mlir/quantization/tensorflow:passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "//tensorflow/compiler/mlir/xla:tf_xla_passes",
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
        ":mhlo_util",
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
        ":mhlo_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
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
        ":transforms",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir:passes",
        "//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_quant_ops",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:compile_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_graph_optimization_pass",
        "//tensorflow/compiler/mlir/xla:legalize_tf",
        "//tensorflow/compiler/mlir/xla:xla_passes",
        "//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:register",
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
    ],
)

tf_cc_binary(
    name = "tf-mhlo-tfl-opt",
    testonly = True,
    tags = ["hostonly"],
    deps = [
        ":mhlo_tfl",
        ":tf_mhlo",
        ":tf_mhlo_tfl",
        ":tf_poly",
        ":tfl_mhlo",
        "//tensorflow/compiler/mlir:passes",
        "//tensorflow/compiler/mlir:tf_mlir_opt_main",
    ],
)

exports_files([
    "run_lit.sh",
])
