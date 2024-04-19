load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

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
    visibility = [
        "//tensorflow/compiler/mlir/lite/stablehlo/odml_converter:__subpackages__",
        "//third_party/odml/infra:__subpackages__",
    ],  # Prototype phase.
    deps = [
        ":all_passes",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/stablehlo:legalize_stablehlo_to_vhlo_pass",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

gentbl_cc_library(
    name = "passes_inc_gen",
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

cc_library(
    name = "all_passes",
    hdrs = ["passes.h"],
    deps = [":passes_inc_gen"],
)

exports_files([
    "run_lit.sh",
])
