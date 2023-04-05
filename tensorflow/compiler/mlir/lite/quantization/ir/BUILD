load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")

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
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:QuantizationOpsTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "QuantOpsIncGen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "QuantOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "QuantOps.cc.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=quantfork",
            ],
            "QuantOpsDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=quantfork",
            ],
            "QuantOpsDialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "QuantOps.td",
    deps = [":QuantizationOpsTdFiles"],
)

gentbl_cc_library(
    name = "QuantPassIncGen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=quantfork",
            ],
            "Passes.h.inc",
        ),
    ],
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
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":QuantOpsIncGen",
        ":QuantPassIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:TransformUtils",
    ],
)
