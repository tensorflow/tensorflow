load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":friends",
    ],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/mlir/lite/...",
    ],
)

tf_cc_binary(
    name = "odml-converter",
    srcs = ["odml_converter_main.cc"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/compiler/mlir/lite/stablehlo/odml_converter:__subpackages__",
        "//third_party/odml/infra:__subpackages__",
    ],  # Prototype phase.
    deps = [
        ":outline_composites",
        ":shlo_simplify",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/stablehlo:legalize_stablehlo_to_vhlo_pass",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Support",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "outline_composites",
    srcs = [
        "transforms/outline_composites.cc",
    ],
    hdrs = ["passes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BufferizationInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "shlo_simplify",
    srcs = [
        "transforms/shlo_simplify.cc",
    ],
    hdrs = ["passes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":folders",
        ":passes_inc_gen",
        ":shlo_simplify_inc_gen",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "shlo_simplify_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_shlo_simplify.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/shlo_simplify.td",
    deps = ["@stablehlo//:stablehlo_ops_td_files"],
)

cc_library(
    name = "folders",
    srcs = ["folders.cc"],
    hdrs = ["folders.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=ODMLConverter",
            ],
            "passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)
