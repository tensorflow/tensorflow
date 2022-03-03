load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = [
        "//tensorflow/compiler/mlir/lite/quantization:__subpackages__",
        "//tensorflow/compiler/mlir/quantization:__subpackages__",
    ],
    licenses = ["notice"],
)

td_library(
    name = "pass_base_td_files",
    srcs = [
        "transforms/passes.td",
    ],
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

td_library(
    name = "quant_ops_td_files",
    srcs = [
        "ir/quant_ops.td",
        "ir/quant_ops_base.td",
    ],
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "quant_ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ir/quant_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ir/quant_ops.cc.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "ir/quant_ops_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "ir/quant_ops_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/quant_ops.td",
    deps = [":quant_ops_td_files"],
)

gentbl_cc_library(
    name = "quant_pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Quant",
            ],
            "transforms/passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/passes.td",
    deps = [":pass_base_td_files"],
)

cc_library(
    name = "quant_ops",
    srcs = [
        "ir/quant_ops.cc",
        "ir/quant_types.cc",
        "ir/type_detail.h",
        "ir/type_parser.cc",
        "transforms/convert_const.cc",
        "transforms/convert_sim_quant.cc",
        "transforms/pass_detail.h",
        "utils/fake_quant_support.cc",
        "utils/quantize_utils.cc",
        "utils/uniform_support.cc",
    ],
    hdrs = [
        "ir/quant_ops.h",
        "ir/quant_types.h",
        "transforms/passes.h",
        "utils/fake_quant_support.h",
        "utils/quantize_utils.h",
        "utils/uniform_support.h",
    ],
    deps = [
        ":quant_ops_inc_gen",
        ":quant_pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:TransformUtils",
    ],
)
