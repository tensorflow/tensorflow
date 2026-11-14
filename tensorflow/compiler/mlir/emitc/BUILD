# EmitC -> EmitC Pass.
# See:
#   https://github.com/llvm/llvm-project/blob/main/mlir/docs/Dialects/EmitC.md

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [
        "//tensorflow/compiler/mlir/emitc/...",
    ],
)

filegroup(
    name = "emitc_ops_td_files",
    srcs = [
        "@llvm-project//mlir:EmitCDialectTdFiles",
    ],
    compatible_with = get_compatible_with_portable(),
)

gentbl_cc_library(
    name = "emitc_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tags = ["emitc"],
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=EmitC",
            ],
            "transforms/passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "passes_header",
    hdrs = [
        "transforms/passes.h",
        "transforms/passes.h.inc",
    ],
    compatible_with = get_compatible_with_portable(),
    tags = ["emitc"],
    deps = [
        "@llvm-project//mlir:EmitCDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "emitc_passes",
    srcs = [
        "emitc_passes.cc",
        "transforms/add_reflection_map.cc"
    ],
    hdrs = [
        "emitc_passes.h",
        "transforms/passes.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":passes_header",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:resource_value_typed_analyzer",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_analysis",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:EmitCDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:EmitCPassIncGen",
        ":emitc_passes_inc_gen",
    ],
)

tf_cc_binary(
    name = "tf-emitc-opt",
    testonly = True,
    srcs = [
        "tf_emitc_opt_main.cc",
    ],
    tags = ["emitc"],
    deps = [
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir:passes",
        "//tensorflow/compiler/mlir:register_common_dialects",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        ":emitc_passes", # buildcleaner:keep
    ],
)

filegroup(
    name = "litfiles",
    srcs = glob(["runlit*py"]),
    visibility = ["//tensorflow:__subpackages__"],
)

