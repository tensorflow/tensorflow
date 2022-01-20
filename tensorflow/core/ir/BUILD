load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_compatible_with = get_compatible_with_cloud(),
    default_visibility = [
        "//tensorflow/compiler/mlir/tensorflow:__subpackages__",
        "//tensorflow/core:__subpackages__",
        "//tensorflow/tools/tfg_graph_transforms:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

td_library(
    name = "InterfacesTdFiles",
    srcs = ["interfaces.td"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "InterfacesIncGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "interfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "interfaces.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "interfaces.td",
    deps = [
        ":InterfacesTdFiles",
    ],
)

# ODS (https://mlir.llvm.org/docs/OpDefinitions/) generation for op and dialect files to include.
td_library(
    name = "DialectTdFiles",
    srcs = [
        "dialect.td",
        "ops.td",
    ],
    deps = [
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "DialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-op-decls",
                "-dialect",
                "tfg",
            ],
            "ops.h.inc",
        ),
        (
            [
                "-gen-op-defs",
                "-dialect",
                "tfg",
            ],
            "ops.cc.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect",
                "tfg",
            ],
            "dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect",
                "tfg",
            ],
            "dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    deps = [
        ":DialectTdFiles",
        ":InterfacesTdFiles",
        "//tensorflow/core/ir/types:DialectTdFiles",
    ],
)

cc_library(
    name = "Dialect",
    srcs = [
        "interfaces.cc",
        "ops.cc",
        "tf_op_wrapper.cc",
        "utility.cc",
    ],
    hdrs = [
        "dialect.h",
        "interfaces.h",
        "ops.h",
        "tf_op_wrapper.h",
        "utility.h",
    ],
    deps = [
        ":DialectIncGen",
        ":InterfacesIncGen",
        "//tensorflow/core/ir/types:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "utility_test",
    size = "small",
    srcs = ["utility_test.cc"],
    deps = [
        ":Dialect",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)
