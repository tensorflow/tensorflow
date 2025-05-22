load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

td_library(
    name = "QuantizationOpsTdFiles",
    srcs = [
        "QuantOps.td",
        "QuantOpsBase.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:QuantizationOpsTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "QuantOpsIncGen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "QuantOps.h.inc": ["-gen-op-decls"],
        "QuantOps.cc.inc": ["-gen-op-defs"],
        "QuantOpsDialect.h.inc": [
            "-gen-dialect-decls",
            "-dialect=quantization",
        ],
        "QuantOpsDialect.cc.inc": [
            "-gen-dialect-defs",
            "-dialect=quantization",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "QuantOps.td",
    deps = [":QuantizationOpsTdFiles"],
)

gentbl_cc_library(
    name = "QuantPassIncGen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"Passes.h.inc": [
        "-gen-pass-decls",
        "-name=tfquant",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "QuantOps",
    srcs = [
        "ConvertConst.cc",
        "ConvertSimQuant.cc",
        "FakeQuantSupport.cc",
        "QuantOps.cc",
        "QuantizeUtils.cc",
        "UniformSupport.cc",
    ],
    hdrs = [
        "FakeQuantSupport.h",
        "Passes.h",
        "QuantOps.h",
        "QuantizeUtils.h",
        "UniformSupport.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":QuantOpsIncGen",
        ":QuantPassIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)
