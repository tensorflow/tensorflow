# Description:
#   The MLIR "Multi-Level Intermediate Representation" Compiler Infrastructure

load("@org_tensorflow//third_party/mlir:tblgen.bzl", "gentbl")

licenses(["notice"])

package(default_visibility = [":friends"])

package_group(
    name = "subpackages",
    packages = ["//..."],
)

package_group(
    name = "friends",
    packages = ["//..."],
)

exports_files([
    "LICENSE.TXT",
    "include/mlir/Dialect/LLVMIR/LLVMOps.td",
    "run_lit.sh",
])

cc_library(
    name = "DialectSymbolRegistry",
    # strip_include_prefix does not apply to textual_hdrs.
    hdrs = ["include/mlir/IR/DialectSymbolRegistry.def"],
    strip_include_prefix = "include/mlir/IR",
    textual_hdrs = ["include/mlir/IR/DialectSymbolRegistry.def"],
)

gentbl(
    name = "OpAsmInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "include/mlir/IR/OpAsmInterface.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "include/mlir/IR/OpAsmInterface.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/IR/OpAsmInterface.td",
    td_srcs = [
        ":OpBaseTdFiles",
    ],
)

cc_library(
    name = "IR",
    srcs = glob([
        "lib/IR/*.cpp",
        "lib/IR/*.h",
    ]),
    hdrs = glob([
        "include/mlir/IR/*.h",
    ]) + [
        "include/mlir/Analysis/CallInterfaces.h",
    ],
    includes = ["include"],
    deps = [
        ":CallOpInterfacesIncGen",
        ":DialectSymbolRegistry",
        ":InferTypeOpInterfaceIncGen",
        ":OpAsmInterfacesIncGen",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "Pass",
    srcs = glob([
        "lib/Pass/*.cpp",
        "lib/Pass/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Pass/*.h",
    ]) + [
        "include/mlir/Analysis/Verifier.h",
    ],
    includes = ["include"],
    linkopts = [
        "-lm",
        "-lpthread",
    ],
    deps = [
        ":IR",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

# TODO(ntv): Update these to enable simplifying the cmake and build files.
cc_library(
    name = "EDSC",
    srcs = [
        "lib/EDSC/Builders.cpp",
        "lib/EDSC/Helpers.cpp",
        "lib/EDSC/Intrinsics.cpp",
    ],
    hdrs = [
        "include/mlir-c/Core.h",
        "include/mlir/EDSC/Builders.h",
        "include/mlir/EDSC/Helpers.h",
        "include/mlir/EDSC/Intrinsics.h",
    ],
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":IR",
        ":LoopOps",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "EDSCInterface",
    srcs = [
        "lib/EDSC/CoreAPIs.cpp",
    ],
    hdrs = [
        "include/mlir-c/Core.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":Parser",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

filegroup(
    name = "OpBaseTdFiles",
    srcs = [
        "include/mlir/IR/OpBase.td",
    ],
)

filegroup(
    name = "AffineOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/AffineOps/AffineOps.td",
        "include/mlir/Dialect/AffineOps/AffineOpsBase.td",
        "include/mlir/Transforms/LoopLikeInterface.td",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "AffineOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/AffineOps/AffineOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/AffineOps/AffineOps.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/AffineOps/AffineOps.td",
    td_srcs = [
        ":AffineOpsTdFiles",
    ],
)

filegroup(
    name = "LoopOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/LoopOps/LoopOps.td",
        "include/mlir/Transforms/LoopLikeInterface.td",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "LoopOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/LoopOps/LoopOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/LoopOps/LoopOps.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LoopOps/LoopOps.td",
    td_srcs = [
        ":LoopOpsTdFiles",
    ],
)

filegroup(
    name = "StdOpsTdFiles",
    srcs = [
        "include/mlir/Analysis/CallInterfaces.td",
        "include/mlir/Dialect/StandardOps/Ops.td",
        "include/mlir/IR/OpAsmInterface.td",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "StandardOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/StandardOps/Ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/StandardOps/Ops.cpp.inc",
        ),
        (
            "-gen-enum-decls",
            "include/mlir/Dialect/StandardOps/OpsEnums.h.inc",
        ),
        (
            "-gen-enum-defs",
            "include/mlir/Dialect/StandardOps/OpsEnums.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/StandardOps/Ops.td",
    td_srcs = [
        ":StdOpsTdFiles",
    ],
)

cc_library(
    name = "Dialect",
    srcs = glob([
        "lib/Dialect/*.cpp",
        "lib/Dialect/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Dialect/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":IR",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "DialectUtils",
    srcs = glob([
        "lib/Dialect/Utils/*.cpp",
        "lib/Dialect/Utils/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Dialect/Utils/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":IR",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "AffineOps",
    srcs = glob(
        [
            "lib/Dialect/AffineOps/*.cpp",
            "lib/Dialect/AffineOps/*.h",
        ],
        exclude = ["lib/Dialect/**/DialectRegistration.cpp"],
    ) + [
        "include/mlir/Transforms/InliningUtils.h",
        "include/mlir/Transforms/LoopLikeInterface.h",
    ],
    hdrs = glob([
        "include/mlir/Dialect/AffineOps/*.h",
    ]) + [
        "include/mlir/Transforms/SideEffectsInterface.h",
    ],
    includes = ["include"],
    deps = [
        ":AffineOpsIncGen",
        ":IR",
        ":LoopLikeOpInterfaceIncGen",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

# Library with affine dialect static initialization.
cc_library(
    name = "AffineDialectRegistration",
    srcs = ["lib/Dialect/AffineOps/DialectRegistration.cpp"],
    deps = [":AffineOps"],
    alwayslink = 1,
)

cc_library(
    name = "AffineToStandardTransforms",
    srcs = glob([
        "lib/Conversion/AffineToStandard/*.cpp",
        "lib/Conversion/AffineToStandard/*.h",
    ]),
    hdrs = glob(["include/mlir/Conversion/AffineToStandard/*.h"]),
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":IR",
        ":LoopOps",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
    ],
    alwayslink = 1,  # contains pass registration
)

# SDBM dialect only contains attribute components that can be constructed given
# a dialect object, so whenever it is used it must also be registered. Therefore
# we don't split out the registration library for it.
cc_library(
    name = "SDBM",
    srcs = glob([
        "lib/Dialect/SDBM/*.cpp",
        "lib/Dialect/SDBM/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Dialect/SDBM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":IR",
        ":Support",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "LoopOps",
    srcs = glob(
        [
            "lib/Dialect/LoopOps/*.cpp",
            "lib/Dialect/LoopOps/*.h",
        ],
        exclude = ["lib/Dialect/**/DialectRegistration.cpp"],
    ),
    hdrs = glob([
        "include/mlir/Dialect/LoopOps/*.h",
    ]) + [
        "include/mlir/Transforms/LoopLikeInterface.h",
        "include/mlir/Transforms/SideEffectsInterface.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LoopLikeOpInterfaceIncGen",
        ":LoopOpsIncGen",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "LoopDialectRegistration",
    srcs = ["lib/Dialect/LoopOps/DialectRegistration.cpp"],
    deps = [":LoopOps"],
    alwayslink = 1,
)

cc_library(
    name = "StandardOps",
    srcs = glob(
        [
            "lib/Dialect/StandardOps/*.cpp",
            "lib/Dialect/StandardOps/*.h",
        ],
        exclude = ["lib/Dialect/**/DialectRegistration.cpp"],
    ),
    hdrs = glob([
        "include/mlir/Dialect/StandardOps/*.h",
    ]) + [
        "include/mlir/Analysis/CallInterfaces.h",
        "include/mlir/Transforms/InliningUtils.h",
    ],
    includes = ["include"],
    deps = [
        ":CallOpInterfacesIncGen",
        ":CommonFolders",
        ":IR",
        ":StandardOpsIncGen",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

# Library with standard dialect static initialization.
cc_library(
    name = "StandardDialectRegistration",
    srcs = ["lib/Dialect/StandardOps/DialectRegistration.cpp"],
    deps = [":StandardOps"],
    alwayslink = 1,
)

cc_library(
    name = "VectorOps",
    srcs = glob(
        [
            "lib/Dialect/VectorOps/*.cpp",
            "lib/Dialect/VectorOps/*.h",
        ],
        exclude = ["lib/Dialect/**/DialectRegistration.cpp"],
    ),
    hdrs = glob([
        "include/mlir/Dialect/VectorOps/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":Analysis",
        ":DialectUtils",
        ":IR",
        ":StandardOps",
        ":Support",
        ":VectorOpsIncGen",
        ":VectorTransformPatternsIncGen",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "VectorDialectRegistration",
    srcs = ["lib/Dialect/VectorOps/DialectRegistration.cpp"],
    deps = [":VectorOps"],
    alwayslink = 1,
)

cc_library(
    name = "Support",
    srcs = glob(
        [
            "lib/Support/*.cpp",
            "lib/Support/*.h",
        ],
        exclude = [
            # TODO(herhut): Move JitRunner out of Support so that Support does not
            # depend on dialect.
            "lib/Support/JitRunner.cpp",
            # TODO(jpienaar): Move this out, else Support depends on Analysis/
            "lib/Support/MlirOptMain.cpp",
            # TODO(jpienaar): Move this out, else Support depends on Analysis/
            "lib/Support/TranslateClParser.cpp",
        ],
    ),
    hdrs = glob([
        "include/mlir/ADT/*.h",
        "include/mlir/Support/*.h",
    ]) + [
        "include/mlir/Translation.h",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "ParserTokenKinds",
    # strip_include_prefix does not apply to textual_hdrs.
    hdrs = ["lib/Parser/TokenKinds.def"],
    strip_include_prefix = "lib/Parser",
    textual_hdrs = ["lib/Parser/TokenKinds.def"],
)

cc_library(
    name = "Parser",
    srcs = glob([
        "lib/Parser/*.cpp",
        "lib/Parser/*.h",
    ]),
    hdrs = glob([
        "include/mlir/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":Analysis",
        ":IR",
        ":ParserTokenKinds",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "LLVMDialect",
    srcs = glob(
        [
            "lib/Dialect/LLVMIR/IR/*.cpp",
            "lib/Dialect/LLVMIR/IR/*.h",
        ],
        exclude = [
            "lib/Dialect/LLVMIR/IR/NVVM*.cpp",
            "lib/Dialect/LLVMIR/IR/NVVM*.h",
            "lib/Dialect/LLVMIR/IR/ROCDL*.cpp",
            "lib/Dialect/LLVMIR/IR/ROCDL*.h",
        ],
    ),
    hdrs = glob(
        [
            "include/mlir/Dialect/LLVMIR/*.h",
        ],
        exclude = [
            "include/mlir/Dialect/LLVMIR/NVVM*.h",
            "include/mlir/Dialect/LLVMIR/ROCDL*.h",
        ],
    ),
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMOpsIncGen",
        ":Support",
        "@llvm-project//llvm:asm_parser",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

filegroup(
    name = "GPUOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/GPU/GPUOps.td",
        "include/mlir/Dialect/LLVMIR/LLVMOpBase.td",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "GPUOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/GPU/GPUOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/GPU/GPUOps.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/GPU/GPUOps.td",
    td_srcs = [
        ":GPUOpsTdFiles",
    ],
)

cc_library(
    name = "GPUDialect",
    srcs = glob(
        [
            "lib/Dialect/GPU/IR/*.cpp",
            "lib/Dialect/GPU/IR/*.h",
        ],
        exclude = ["lib/Dialect/**/DialectRegistration.cpp"],
    ),
    hdrs = glob([
        "include/mlir/Dialect/GPU/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":GPUOpsIncGen",
        ":IR",
        ":LLVMDialect",
        ":StandardOps",
    ],
)

cc_library(
    name = "GPUDialectRegistration",
    srcs = ["lib/Dialect/GPU/IR/DialectRegistration.cpp"],
    deps = [
        ":GPUDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "GPUTransforms",
    srcs = glob(
        [
            "lib/Dialect/GPU/Transforms/*.cpp",
            "lib/Dialect/GPU/Transforms/*.h",
        ],
        exclude = ["lib/Dialect/**/DialectRegistration.cpp"],
    ),
    hdrs = ["include/mlir/Dialect/GPU/Passes.h"],
    includes = ["include"],
    deps = [
        ":EDSC",
        ":GPUDialect",
        ":GPUDialectRegistration",
        ":IR",
        ":LoopOps",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
    ],
    alwayslink = 1,
)

filegroup(
    name = "LLVMOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/LLVMIR/LLVMOpBase.td",
        "include/mlir/Dialect/LLVMIR/LLVMOps.td",
        ":OpBaseTdFiles",
    ],
)

cc_library(
    name = "GPUCommonTransforms",
    hdrs = [
        "lib/Conversion/GPUCommon/IndexIntrinsicsOpLowering.h",
        "lib/Conversion/GPUCommon/OpToFuncCallLowering.h",
    ],
    deps = [
        ":GPUDialect",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":StandardOps",
        "@llvm-project//llvm:support",
    ],
)

gentbl(
    name = "GPUToNVVMGen",
    strip_include_prefix = "lib/Conversion/GPUToNVVM",
    tbl_outs = [
        (
            "-gen-rewriters",
            "lib/Conversion/GPUToNVVM/GPUToNVVM.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "lib/Conversion/GPUToNVVM/GPUToNVVM.td",
    td_srcs = [
        ":GPUOpsTdFiles",
        ":NVVMOpsTdFiles",
    ],
)

cc_library(
    name = "GPUToNVVMTransforms",
    srcs = glob([
        "lib/Conversion/GPUToNVVM/*.cpp",
        "lib/Conversion/GPUToNVVM/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Conversion/GPUToNVVM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":GPUCommonTransforms",
        ":GPUDialect",
        ":GPUToNVVMGen",
        ":IR",
        ":LLVMTransforms",
        ":NVVMDialect",
        ":Pass",
        ":Transforms",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "GPUToROCDLTransforms",
    srcs = ["lib/Conversion/GPUToROCDL/LowerGpuOpsToROCDLOps.cpp"],
    hdrs = [
        "include/mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h",
    ],
    includes = ["include"],
    deps = [
        ":GPUCommonTransforms",
        ":GPUDialect",
        ":LLVMTransforms",
        ":Pass",
        ":ROCDLDialect",
        ":Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "GPUToCUDATransforms",
    srcs = [
        "lib/Conversion/GPUToCUDA/ConvertKernelFuncToCubin.cpp",
        "lib/Conversion/GPUToCUDA/ConvertLaunchFuncToCudaCalls.cpp",
    ],
    hdrs = ["include/mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"],
    includes = ["include"],
    deps = [
        ":GPUDialect",
        ":IR",
        ":LLVMDialect",
        ":Pass",
        ":Support",
        ":TargetNVVMIR",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:nvptx_target",  # buildcleaner: keep
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:target",
    ],
    alwayslink = 1,
)

gentbl(
    name = "GPUToSPIRVIncGen",
    strip_include_prefix = "lib/Conversion/GPUToSPIRV",
    tbl_outs = [
        (
            "-gen-rewriters",
            "lib/Conversion/GPUToSPIRV/GPUToSPIRV.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "lib/Conversion/GPUToSPIRV/GPUToSPIRV.td",
    td_srcs = [
        ":GPUOpsTdFiles",
        ":SPIRVOpsTdFiles",
    ],
)

cc_library(
    name = "GPUToSPIRVTransforms",
    srcs = [
        "lib/Conversion/GPUToSPIRV/ConvertGPUToSPIRV.cpp",
        "lib/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.cpp",
    ],
    hdrs = [
        "include/mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRV.h",
        "include/mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h",
    ],
    includes = [
        "include",
        "lib/Conversions/GPUToSPIRV",
    ],
    deps = [
        ":GPUDialect",
        ":GPUToSPIRVIncGen",
        ":IR",
        ":LoopOps",
        ":Pass",
        ":SPIRVDialect",
        ":SPIRVLowering",
        ":StandardToSPIRVConversions",
        ":Support",
        ":Transforms",
    ],
    alwayslink = 1,
)

gentbl(
    name = "LLVMOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/LLVMIR/LLVMOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/LLVMIR/LLVMOps.cpp.inc",
        ),
        (
            "-gen-enum-decls",
            "include/mlir/Dialect/LLVMIR/LLVMOpsEnums.h.inc",
        ),
        (
            "-gen-enum-defs",
            "include/mlir/Dialect/LLVMIR/LLVMOpsEnums.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/LLVMOps.td",
    td_srcs = [
        ":LLVMOpsTdFiles",
    ],
)

gentbl(
    name = "LLVMConversionIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-llvmir-conversions",
            "include/mlir/Dialect/LLVMIR/LLVMConversions.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/LLVMOps.td",
    td_srcs = [
        ":LLVMOpsTdFiles",
    ],
)

cc_library(
    name = "NVVMDialect",
    srcs = [
        "lib/Dialect/LLVMIR/IR/NVVMDialect.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/LLVMIR/NVVMDialect.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":NVVMOpsIncGen",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:asm_parser",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

filegroup(
    name = "NVVMOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/LLVMIR/LLVMOpBase.td",
        "include/mlir/Dialect/LLVMIR/NVVMOps.td",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "NVVMOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/LLVMIR/NVVMOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/LLVMIR/NVVMOps.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/NVVMOps.td",
    td_srcs = [
        ":NVVMOpsTdFiles",
    ],
)

gentbl(
    name = "NVVMConversionIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-llvmir-conversions",
            "include/mlir/Dialect/LLVMIR/NVVMConversions.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/NVVMOps.td",
    td_srcs = [
        ":NVVMOpsTdFiles",
    ],
)

cc_library(
    name = "ROCDLDialect",
    srcs = [
        "lib/Dialect/LLVMIR/IR/ROCDLDialect.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/LLVMIR/ROCDLDialect.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":ROCDLOpsIncGen",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:asm_parser",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

filegroup(
    name = "ROCDLOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/LLVMIR/LLVMOpBase.td",
        "include/mlir/Dialect/LLVMIR/ROCDLOps.td",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "ROCDLOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/LLVMIR/ROCDLOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/LLVMIR/ROCDLOps.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/ROCDLOps.td",
    td_srcs = [
        ":ROCDLOpsTdFiles",
    ],
)

gentbl(
    name = "ROCDLConversionIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-llvmir-conversions",
            "include/mlir/Dialect/LLVMIR/ROCDLConversions.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/ROCDLOps.td",
    td_srcs = [
        ":ROCDLOpsTdFiles",
    ],
)

# TODO(gcmn): Update SPIRV dependencies so that they map better to cmake files.
filegroup(
    name = "SPIRVOpsTdFiles",
    srcs = [
        "include/mlir/Analysis/CallInterfaces.td",
        ":OpBaseTdFiles",
    ] + glob(["include/mlir/Dialect/SPIRV/*.td"]),
)

gentbl(
    name = "SPIRVOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/SPIRV/SPIRVOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/SPIRV/SPIRVOps.cpp.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/Dialects/SPIRV/SPIRVOps.md",
        ),
        (
            "-gen-enum-decls",
            "include/mlir/Dialect/SPIRV/SPIRVEnums.h.inc",
        ),
        (
            "-gen-enum-defs",
            "include/mlir/Dialect/SPIRV/SPIRVEnums.cpp.inc",
        ),
        (
            "-gen-spirv-enum-avail-decls",
            "include/mlir/Dialect/SPIRV/SPIRVEnumAvailability.h.inc",
        ),
        (
            "-gen-spirv-enum-avail-defs",
            "include/mlir/Dialect/SPIRV/SPIRVEnumAvailability.cpp.inc",
        ),
        (
            "-gen-spirv-capability-implication",
            "include/mlir/Dialect/SPIRV/SPIRVCapabilityImplication.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/SPIRVOps.td",
    td_srcs = [
        ":SPIRVOpsTdFiles",
    ],
)

gentbl(
    name = "SPIRVCanonicalizationIncGen",
    strip_include_prefix = "lib/Dialect/SPIRV",
    tbl_outs = [
        (
            "-gen-rewriters",
            "lib/Dialect/SPIRV/SPIRVCanonicalization.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "lib/Dialect/SPIRV/SPIRVCanonicalization.td",
    td_srcs = [
        ":SPIRVOpsTdFiles",
        "lib/Dialect/SPIRV/SPIRVCanonicalization.td",
    ],
)

gentbl(
    name = "StandardToSPIRVGen",
    strip_include_prefix = "lib/Conversion/StandardToSPIRV",
    tbl_outs = [
        (
            "-gen-rewriters",
            "lib/Conversion/StandardToSPIRV/StandardToSPIRV.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "lib/Conversion/StandardToSPIRV/StandardToSPIRV.td",
    td_srcs = [
        ":SPIRVOpsTdFiles",
        ":StdOpsTdFiles",
    ],
)

gentbl(
    name = "SPIRVAvailabilityIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-avail-interface-decls",
            "include/mlir/Dialect/SPIRV/SPIRVAvailability.h.inc",
        ),
        (
            "-gen-avail-interface-defs",
            "include/mlir/Dialect/SPIRV/SPIRVAvailability.cpp.inc",
        ),
        (
            "-gen-spirv-avail-impls",
            "include/mlir/Dialect/SPIRV/SPIRVOpAvailabilityImpl.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/SPIRVOps.td",
    td_srcs = [
        ":SPIRVOpsTdFiles",
    ],
)

gentbl(
    name = "SPIRVTargetAndABIStructGen",
    tbl_outs = [
        (
            "-gen-struct-attr-decls",
            "include/mlir/Dialect/SPIRV/TargetAndABI.h.inc",
        ),
        (
            "-gen-struct-attr-defs",
            "include/mlir/Dialect/SPIRV/TargetAndABI.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/TargetAndABI.td",
    td_srcs = [
        ":SPIRVOpsTdFiles",
        ":StdOpsTdFiles",
    ],
)

gentbl(
    name = "SPIRVOpUtilsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-spirv-op-utils",
            "include/mlir/Dialect/SPIRV/SPIRVOpUtils.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/SPIRVBase.td",
    td_srcs = [
        ":SPIRVOpsTdFiles",
        ":SPIRVAvailabilityIncGen",
    ],
)

gentbl(
    name = "SPIRVSerializationGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-spirv-serialization",
            "include/mlir/Dialect/SPIRV/SPIRVSerialization.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/SPIRVOps.td",
    td_srcs = [
        ":SPIRVOpsTdFiles",
    ],
)

cc_library(
    name = "SPIRVDialect",
    srcs = glob(
        [
            "lib/Dialect/SPIRV/*.cpp",
            "lib/Dialect/SPIRV/*.h",
        ],
        exclude = [
            "lib/Dialect/**/DialectRegistration.cpp",
            "lib/Dialect/SPIRV/SPIRVLowering.cpp",
        ],
    ) + [
        "include/mlir/Transforms/InliningUtils.h",
    ],
    hdrs = glob(
        [
            "include/mlir/Dialect/SPIRV/*.h",
        ],
        exclude = [
            "include/mlir/Dialect/SPIRV/SPIRVBinaryUtils.h",
            "include/mlir/Dialect/SPIRV/SPIRVLowering.h",
        ],
    ),
    includes = ["include"],
    deps = [
        ":CommonFolders",
        ":IR",
        ":Parser",
        ":Pass",
        ":SPIRVAvailabilityIncGen",
        ":SPIRVCanonicalizationIncGen",
        ":SPIRVOpUtilsIncGen",
        ":SPIRVOpsIncGen",
        ":SPIRVSerializationGen",
        ":SPIRVTargetAndABIStructGen",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "SPIRVLowering",
    srcs = [
        "lib/Dialect/SPIRV/SPIRVLowering.cpp",
        "lib/Dialect/SPIRV/Transforms/DecorateSPIRVCompositeTypeLayoutPass.cpp",
        "lib/Dialect/SPIRV/Transforms/LowerABIAttributesPass.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/SPIRV/Passes.h",
        "include/mlir/Dialect/SPIRV/SPIRVLowering.h",
        "include/mlir/Dialect/SPIRV/TargetAndABI.h",
    ],
    includes = [
        "include",
    ],
    deps = [
        ":IR",
        ":Pass",
        ":SPIRVDialect",
        ":SPIRVTargetAndABIStructGen",
        ":StandardOps",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "StandardToSPIRVConversions",
    srcs = glob([
        "lib/Conversion/StandardToSPIRV/*.cpp",
        "lib/Conversion/StandardToSPIRV/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Conversion/StandardToSPIRV/*.h",
    ]),
    includes = [
        "include",
        "lib/Conversion/StandardToSPIRV",
    ],
    deps = [
        ":IR",
        ":Pass",
        ":SPIRVDialect",
        ":SPIRVLowering",
        ":StandardOps",
        ":StandardToSPIRVGen",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "SPIRVSerialization",
    srcs = glob(
        [
            "lib/Dialect/SPIRV/Serialization/*.cpp",
        ],
        exclude = [
            "lib/Dialect/SPIRV/Serialization/TranslateRegistration.cpp",
        ],
    ),
    hdrs = [
        "include/mlir/Dialect/SPIRV/SPIRVBinaryUtils.h",
        "include/mlir/Dialect/SPIRV/Serialization.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":SPIRVDialect",
        ":SPIRVOpUtilsIncGen",
        ":SPIRVOpsIncGen",
        ":SPIRVSerializationGen",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "SPIRVTranslateRegistration",
    srcs = [
        "lib/Dialect/SPIRV/Serialization/TranslateRegistration.cpp",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":Parser",
        ":SPIRVDialect",
        ":SPIRVSerialization",
        ":Support",
        ":Translation",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "SPIRVDialectRegistration",
    srcs = ["lib/Dialect/SPIRV/DialectRegistration.cpp"],
    deps = [
        ":SPIRVDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "TransformUtils",
    srcs = glob([
        "lib/Transforms/Utils/*.cpp",
        "lib/Transforms/Utils/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Transforms/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":Analysis",
        ":IR",
        ":LoopLikeOpInterfaceIncGen",
        ":LoopOps",
        ":StandardDialectRegistration",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

gentbl(
    name = "LoopLikeOpInterfaceIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "include/mlir/Transforms/LoopLikeInterface.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "include/mlir/Transforms/LoopLikeInterface.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Transforms/LoopLikeInterface.td",
    td_srcs = [
        ":OpBaseTdFiles",
    ],
)

cc_library(
    name = "Transforms",
    srcs = glob([
        "lib/Transforms/*.cpp",
        "lib/Transforms/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Transforms/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":Analysis",
        ":EDSC",
        ":IR",
        ":LoopLikeOpInterfaceIncGen",
        ":LoopOps",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":VectorAnalysis",
        ":VectorOps",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "CommonFolders",
    srcs = [
    ],
    hdrs = [
        "include/mlir/Dialect/CommonFolders.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "LoopsToGPU",
    srcs = [
        "lib/Conversion/LoopsToGPU/LoopsToGPU.cpp",
    ],
    hdrs = [
        "include/mlir/Conversion/LoopsToGPU/LoopsToGPU.h",
    ],
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":AffineToStandardTransforms",
        ":GPUDialect",
        ":IR",
        ":LinalgTransforms",
        ":LoopOps",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":Transforms",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "LoopsToGPUPass",
    srcs = [
        "lib/Conversion/LoopsToGPU/LoopsToGPUPass.cpp",
    ],
    hdrs = [
        "include/mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h",
    ],
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":LoopOps",
        ":LoopsToGPU",
        ":Pass",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "CFGTransforms",
    srcs = [
        "lib/Conversion/LoopToStandard/ConvertLoopToStandard.cpp",
    ],
    hdrs = [
        "include/mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":LoopOps",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "LLVMTransforms",
    srcs = [
        "lib/Conversion/StandardToLLVM/ConvertStandardToLLVM.cpp",
    ],
    hdrs = [
        "include/mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h",
        "include/mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h",
    ],
    includes = ["include"],
    deps = [
        ":CFGTransforms",
        ":IR",
        ":LLVMDialect",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":Transforms",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

gentbl(
    name = "CallOpInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "include/mlir/Analysis/CallInterfaces.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "include/mlir/Analysis/CallInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Analysis/CallInterfaces.td",
    td_srcs = [
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "InferTypeOpInterfaceIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "include/mlir/Analysis/InferTypeOpInterface.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "include/mlir/Analysis/InferTypeOpInterface.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Analysis/InferTypeOpInterface.td",
    td_srcs = [
        ":OpBaseTdFiles",
    ],
)

cc_library(
    name = "Analysis",
    srcs = glob(
        [
            "lib/Analysis/*.cpp",
            "lib/Analysis/*.h",
        ],
        exclude = [
            "lib/Analysis/Vector*.cpp",
            "lib/Analysis/Vector*.h",
        ],
    ),
    hdrs = glob(
        [
            "include/mlir/Analysis/*.h",
        ],
        exclude = [
            "include/mlir/Analysis/Vector*.h",
        ],
    ),
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":CallOpInterfacesIncGen",
        ":IR",
        ":InferTypeOpInterfaceIncGen",
        ":LoopOps",
        ":Pass",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "VectorAnalysis",
    srcs = glob([
        "lib/Analysis/Vector*.cpp",
        "lib/Analysis/Vector*.h",
    ]),
    hdrs = glob([
        "include/mlir/Analysis/Vector*.h",
    ]),
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":Analysis",
        ":IR",
        ":StandardOps",
        ":Support",
        ":VectorOps",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "Translation",
    srcs = glob([
        "lib/Translation/*.cpp",
        "lib/Translation/*.h",
    ]),
    hdrs = glob([
        "include/mlir/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":IR",
        ":Parser",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "LLVMIRModuleTranslation",
    srcs = [
        "lib/Target/LLVMIR/ModuleTranslation.cpp",
    ],
    hdrs = [
        "include/mlir/Target/LLVMIR/ModuleTranslation.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMConversionIncGen",
        ":LLVMDialect",
        ":Support",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:transform_utils",
    ],
)

cc_library(
    name = "TargetLLVMIR",
    srcs = [
        "lib/Target/LLVMIR/ConvertFromLLVMIR.cpp",
        "lib/Target/LLVMIR/ConvertToLLVMIR.cpp",
    ],
    hdrs = [
        "include/mlir/Target/LLVMIR.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":LLVMIRModuleTranslation",
        ":Support",
        ":Translation",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:ir_reader",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "TargetNVVMIR",
    srcs = [
        "lib/Target/LLVMIR/ConvertToNVVMIR.cpp",
    ],
    hdrs = [
        "include/mlir/Target/NVVMIR.h",
    ],
    includes = ["include"],
    deps = [
        ":GPUDialect",
        ":IR",
        ":LLVMDialect",
        ":LLVMIRModuleTranslation",
        ":NVVMConversionIncGen",
        ":NVVMDialect",
        ":Support",
        ":Translation",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "TargetROCDLIR",
    srcs = [
        "lib/Target/LLVMIR/ConvertToROCDLIR.cpp",
    ],
    hdrs = [
        "include/mlir/Target/ROCDLIR.h",
    ],
    includes = ["include"],
    deps = [
        ":GPUDialect",
        ":IR",
        ":LLVMDialect",
        ":LLVMIRModuleTranslation",
        ":ROCDLConversionIncGen",
        ":ROCDLDialect",
        ":Support",
        ":Translation",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

# TODO(zinenko): Update these so that we can simplify mapping to cmake.
cc_library(
    name = "ExecutionEngine",
    srcs = [
        "lib/ExecutionEngine/ExecutionEngine.cpp",
    ],
    hdrs = [
        "include/mlir/ExecutionEngine/ExecutionEngine.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":Support",
        ":TargetLLVMIR",
        ":Translation",
        "@llvm-project//llvm:bit_reader",
        "@llvm-project//llvm:bit_writer",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:execution_engine",
        "@llvm-project//llvm:mc",
        "@llvm-project//llvm:orc_jit",
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:target",  # fixdeps: keep
        "@llvm-project//llvm:transform_utils",
        "@llvm-project//llvm:x86_code_gen",  # fixdeps: keep
        "@llvm-project//llvm:x86_disassembler",  # fixdeps: keep
    ],
)

cc_library(
    name = "ExecutionEngineUtils",
    srcs = [
        "lib/ExecutionEngine/OptUtils.cpp",
    ],
    hdrs = [
        "include/mlir/ExecutionEngine/OptUtils.h",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//llvm:analysis",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:ipo",
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:target",
    ],
)

# TODO(jpienaar): Update this.
cc_library(
    name = "MlirOptLib",
    srcs = [
        "lib/Support/MlirOptMain.cpp",
    ],
    hdrs = [
        "include/mlir/Analysis/Passes.h",
    ],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":GPUToNVVMTransforms",
        ":GPUToROCDLTransforms",
        ":GPUToSPIRVTransforms",
        ":GPUTransforms",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":LinalgToLLVM",
        ":NVVMDialect",
        ":Parser",
        ":Pass",
        ":QuantizerTransforms",
        ":SPIRVDialectRegistration",
        ":StandardToSPIRVConversions",
        ":Support",
        ":Transforms",
        ":VectorToLLVM",
        ":VectorToLoops",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "TranslateClParser",
    srcs = ["lib/Support/TranslateClParser.cpp"],
    hdrs = ["include/mlir/Support/TranslateClParser.h"],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":IR",
        ":Parser",
        ":Support",
        ":Translation",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "MlirTranslateMain",
    srcs = ["tools/mlir-translate/mlir-translate.cpp"],
    deps = [
        ":IR",
        ":Parser",
        ":Support",
        ":TranslateClParser",
        ":Translation",
        "@llvm-project//llvm:support",
    ],
)

cc_binary(
    name = "mlir-translate",
    deps = [
        ":LoopDialectRegistration",
        ":MlirTranslateMain",
        ":SPIRVDialectRegistration",
        ":SPIRVTranslateRegistration",
        ":StandardDialectRegistration",
        ":TargetLLVMIR",
        ":TargetNVVMIR",
        ":TargetROCDLIR",
        ":VectorDialectRegistration",
    ],
)

# TODO(jpienaar): This library should be removed.
cc_library(
    name = "MlirOptMain",
    srcs = [
        "tools/mlir-opt/mlir-opt.cpp",
    ],
    deps = [
        ":Analysis",
        ":MlirOptLib",
        ":Pass",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_binary(
    name = "mlir-opt",
    deps = [
        ":AffineDialectRegistration",
        ":Analysis",
        ":FxpMathOps",
        ":FxpMathOpsDialectRegistration",
        ":GPUDialectRegistration",
        ":IR",
        ":LinalgDialectRegistration",
        ":LoopDialectRegistration",
        ":LoopsToGPUPass",
        ":MlirOptLib",
        ":MlirOptMain",
        ":OpenMPDialect",
        ":QuantOps",
        ":QuantOpsDialectRegistration",
        ":ROCDLDialect",
        ":StandardDialectRegistration",
        ":Transforms",
        ":VectorDialectRegistration",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir/test:TestDialect",
        "@llvm-project//mlir/test:TestIR",
        "@llvm-project//mlir/test:TestPass",
        "@llvm-project//mlir/test:TestTransforms",
        "@llvm-project//mlir/test/Dialect/SPIRV:TestPasses",
    ],
)

cc_library(
    name = "MlirJitRunner",
    srcs = ["lib/Support/JitRunner.cpp"],
    hdrs = ["include/mlir/Support/JitRunner.h"],
    includes = ["include"],
    deps = [
        ":AffineDialectRegistration",
        ":CFGTransforms",
        ":ExecutionEngine",
        ":ExecutionEngineUtils",
        ":IR",
        ":LLVMDialect",
        ":Parser",
        ":Pass",
        ":Support",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:orc_jit",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "mlir-cpu-runner",
    srcs = ["tools/mlir-cpu-runner/mlir-cpu-runner.cpp"],
    linkopts = ["-ldl"],
    deps = [":MlirJitRunner"],
)

cc_binary(
    name = "tools/libcuda-runtime-wrappers.so",
    srcs = ["tools/mlir-cuda-runner/cuda-runtime-wrappers.cpp"],
    linkshared = True,
    deps = [
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cuda_runtime",
        "//third_party/gpus/cuda:libcuda",
        "@llvm-project//llvm:support",
    ],
)

cc_binary(
    name = "mlir-cuda-runner",
    srcs = ["tools/mlir-cuda-runner/mlir-cuda-runner.cpp"],
    data = [
        ":tools/libcuda-runtime-wrappers.so",
        "@llvm-project//mlir/test/mlir-cpu-runner:libmlir_runner_utils.so",
    ],
    deps = [
        ":GPUDialect",
        ":GPUDialectRegistration",
        ":GPUToNVVMTransforms",
        ":GPUToROCDLTransforms",
        ":GPUTransforms",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":MlirJitRunner",
        ":NVVMDialect",
        ":Pass",
        ":Transforms",
        "//devtools/build/runtime:get_runfiles_dir",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cuda_runtime",
        "//third_party/gpus/cuda:libcuda",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "TableGen",
    srcs = glob(["lib/TableGen/*.cpp"]),
    hdrs = glob(["include/mlir/TableGen/*.h"]),
    includes = ["include"],
    deps = [
        ":Support",
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:tablegen",
    ],
)

cc_library(
    name = "MlirTableGenMain",
    srcs = [
        "tools/mlir-tblgen/mlir-tblgen.cpp",
    ],
    includes = ["include"],
    deps = [
        ":Support",
        ":TableGen",
        "@llvm-project//llvm:config",
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:tablegen",
    ],
)

cc_binary(
    name = "mlir-tblgen",
    srcs = glob([
        "tools/mlir-tblgen/*.h",
        "tools/mlir-tblgen/*.cpp",
    ]),
    linkopts = [
        "-lm",
        "-lpthread",
    ],
    deps = [
        ":MlirTableGenMain",
        ":Support",
        ":TableGen",
        "@llvm-project//llvm:config",
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:tablegen",
    ],
)

## OpenMP dialect
gentbl(
    name = "OpenMPOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/OpenMP/OpenMPOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/OpenMP/OpenMPOps.cpp.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/Dialects/OpenMP/OpenMPOps.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/OpenMP/OpenMPOps.td",
    td_srcs = [
        ":OpBaseTdFiles",
    ],
)

cc_library(
    name = "OpenMPDialect",
    srcs = glob(
        [
            "lib/Dialect/OpenMP/IR/*.cpp",
            "lib/Dialect/OpenMP/IR/*.h",
        ],
    ),
    hdrs = glob([
        "include/mlir/Dialect/OpenMP/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":IR",
        ":OpenMPOpsIncGen",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

filegroup(
    name = "QuantizationOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/QuantOps/QuantOps.td",
        "include/mlir/Dialect/QuantOps/QuantPredicates.td",
        ":OpBaseTdFiles",
    ],
)

## QuantOps dialect
gentbl(
    name = "QuantOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/QuantOps/QuantOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/QuantOps/QuantOps.cpp.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/Dialects/QuantOps/QuantOps.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/QuantOps/QuantOps.td",
    td_srcs = [
        ":QuantizationOpsTdFiles",
    ],
)

cc_library(
    name = "QuantOps",
    srcs = [
        "lib/Dialect/QuantOps/IR/QuantOps.cpp",
        "lib/Dialect/QuantOps/IR/QuantTypes.cpp",
        "lib/Dialect/QuantOps/IR/TypeDetail.h",
        "lib/Dialect/QuantOps/IR/TypeParser.cpp",
        "lib/Dialect/QuantOps/Transforms/ConvertConst.cpp",
        "lib/Dialect/QuantOps/Transforms/ConvertSimQuant.cpp",
        "lib/Dialect/QuantOps/Utils/FakeQuantSupport.cpp",
        "lib/Dialect/QuantOps/Utils/QuantizeUtils.cpp",
        "lib/Dialect/QuantOps/Utils/UniformSupport.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/QuantOps/FakeQuantSupport.h",
        "include/mlir/Dialect/QuantOps/Passes.h",
        "include/mlir/Dialect/QuantOps/QuantOps.h",
        "include/mlir/Dialect/QuantOps/QuantTypes.h",
        "include/mlir/Dialect/QuantOps/QuantizeUtils.h",
        "include/mlir/Dialect/QuantOps/UniformSupport.h",
    ],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":IR",
        ":Pass",
        ":QuantOpsIncGen",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "QuantOpsDialectRegistration",
    srcs = ["lib/Dialect/QuantOps/IR/DialectRegistration.cpp"],
    deps = [":QuantOps"],
    alwayslink = 1,
)

filegroup(
    name = "FxpMathOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/FxpMathOps/FxpMathOps.td",
        "include/mlir/Dialect/QuantOps/QuantPredicates.td",
        ":OpBaseTdFiles",
    ],
)

## FxpMathOps dialect
gentbl(
    name = "FxpMathOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/FxpMathOps/FxpMathOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/FxpMathOps/FxpMathOps.cpp.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/Dialects/FxpMathOps/FxpMathOps.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/FxpMathOps/FxpMathOps.td",
    td_srcs = [
        ":FxpMathOpsTdFiles",
    ],
)

cc_library(
    name = "FxpMathOps",
    srcs = [
        "lib/Dialect/FxpMathOps/IR/FxpMathOps.cpp",
        "lib/Dialect/FxpMathOps/Transforms/LowerUniformRealMath.cpp",
        "lib/Dialect/FxpMathOps/Transforms/UniformKernelUtils.h",
    ],
    hdrs = [
        "include/mlir/Dialect/FxpMathOps/FxpMathOps.h",
        "include/mlir/Dialect/FxpMathOps/Passes.h",
    ],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":FxpMathOpsIncGen",
        ":IR",
        ":Pass",
        ":QuantOps",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "FxpMathOpsDialectRegistration",
    srcs = ["lib/Dialect/FxpMathOps/IR/DialectRegistration.cpp"],
    deps = [":FxpMathOps"],
    alwayslink = 1,
)

filegroup(
    name = "LinalgOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Linalg/IR/LinalgBase.td",
        "include/mlir/Dialect/Linalg/IR/LinalgOps.td",
        "include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td",
        ":AffineOpsTdFiles",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "LinalgOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/Linalg/IR/LinalgOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/IR/LinalgOps.td",
    td_srcs = [
        ":LinalgOpsTdFiles",
    ],
)

filegroup(
    name = "LinalgStructuredOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Linalg/IR/LinalgBase.td",
        "include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td",
        ":AffineOpsTdFiles",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "LinalgStructuredOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc",
        ),
        (
            "-gen-op-interface-decls",
            "include/mlir/Dialect/Linalg/IR/LinalgStructuredOpsInterfaces.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "include/mlir/Dialect/Linalg/IR/LinalgStructuredOpsInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td",
    td_srcs = [
        ":LinalgStructuredOpsTdFiles",
    ],
)

filegroup(
    name = "LinalgDocTdFiles",
    srcs = [
        "include/mlir/Dialect/Linalg/IR/LinalgDoc.td",
        ":LinalgOpsTdFiles",
    ],
)

gentbl(
    name = "LinalgDocIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-doc",
            "g3doc/Dialects/Linalg/LinalgOps.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/IR/LinalgDoc.td",
    td_srcs = [
        ":LinalgDocTdFiles",
    ],
)

filegroup(
    name = "LinalgTransformPatternsTdFiles",
    srcs = [
        "include/mlir/Dialect/Linalg/Transforms/LinalgTransformPatterns.td",
        ":AffineOpsTdFiles",
        ":LinalgOpsTdFiles",
        ":LinalgStructuredOpsTdFiles",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "LinalgTransformPatternsIncGen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "include/mlir/Dialect/Linalg/Transforms/LinalgTransformPatterns.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/Transforms/LinalgTransformPatterns.td",
    td_srcs = [
        ":LinalgTransformPatternsTdFiles",
    ],
)

cc_library(
    name = "LinalgToLLVM",
    srcs = glob([
        "lib/Conversion/LinalgToLLVM/*.cpp",
        "lib/Conversion/LinalgToLLVM/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Conversion/LinalgToLLVM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":AffineToStandardTransforms",
        ":Analysis",
        ":CFGTransforms",
        ":EDSC",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":LinalgOps",
        ":LinalgTransforms",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
        ":VectorToLLVM",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "LinalgOps",
    srcs = [
        "lib/Dialect/Linalg/IR/LinalgOps.cpp",
        "lib/Dialect/Linalg/IR/LinalgTypes.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/Linalg/IR/LinalgOps.h",
        "include/mlir/Dialect/Linalg/IR/LinalgTraits.h",
        "include/mlir/Dialect/Linalg/IR/LinalgTypes.h",
    ],
    includes = ["include"],
    deps = [
        ":DialectUtils",
        ":IR",
        ":LinalgOpsIncGen",
        ":LinalgStructuredOpsIncGen",
        ":Parser",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "LinalgTransforms",
    srcs = [
        "lib/Dialect/Linalg/Analysis/DependenceAnalysis.cpp",
        "lib/Dialect/Linalg/EDSC/Builders.cpp",
        "lib/Dialect/Linalg/Transforms/Fusion.cpp",
        "lib/Dialect/Linalg/Transforms/LinalgToLoops.cpp",
        "lib/Dialect/Linalg/Transforms/LinalgTransforms.cpp",
        "lib/Dialect/Linalg/Transforms/Promotion.cpp",
        "lib/Dialect/Linalg/Transforms/Tiling.cpp",
        "lib/Dialect/Linalg/Utils/Utils.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h",
        "include/mlir/Dialect/Linalg/EDSC/Builders.h",
        "include/mlir/Dialect/Linalg/EDSC/Intrinsics.h",
        "include/mlir/Dialect/Linalg/Passes.h",
        "include/mlir/Dialect/Linalg/Transforms/LinalgTransforms.h",
        "include/mlir/Dialect/Linalg/Utils/Utils.h",
    ],
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":AffineToStandardTransforms",
        ":Analysis",
        ":CFGTransforms",
        ":DialectUtils",
        ":EDSC",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":LinalgOps",
        ":LinalgOpsIncGen",
        ":LinalgStructuredOpsIncGen",
        ":LinalgTransformPatternsIncGen",
        ":LoopOps",
        ":Parser",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":Transforms",
        ":VectorOps",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "LinalgDialectRegistration",
    srcs = ["lib/Dialect/Linalg/IR/LinalgRegistration.cpp"],
    deps = [":LinalgOps"],
    alwayslink = 1,
)

cc_library(
    name = "QuantizerSupportLib",
    srcs = glob([
        "lib/Quantizer/Configurations/*.cpp",
        "lib/Quantizer/Support/*.cpp",
        "lib/Quantizer/Configurations/*.h",
        "lib/Quantizer/Support/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Quantizer/Configurations/*.h",
        "include/mlir/Quantizer/Support/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":FxpMathOps",
        ":IR",
        ":QuantOps",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "QuantizerTransforms",
    srcs = glob([
        "lib/Quantizer/Transforms/*.cpp",
        "lib/Quantizer/Transforms/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Quantizer/Transforms/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":IR",
        ":Pass",
        ":QuantOps",
        ":QuantizerSupportLib",
        ":Support",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

filegroup(
    name = "VectorOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/VectorOps/VectorOps.td",
        ":AffineOpsTdFiles",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "VectorOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/VectorOps/VectorOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/VectorOps/VectorOps.cpp.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/Dialects/Vector/VectorOps.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/VectorOps/VectorOps.td",
    td_srcs = [
        ":VectorOpsTdFiles",
    ],
)

filegroup(
    name = "VectorTransformPatternsTdFiles",
    srcs = [
        "include/mlir/Dialect/VectorOps/VectorTransformPatterns.td",
        ":AffineOpsTdFiles",
        ":LinalgOpsTdFiles",
        ":LinalgStructuredOpsTdFiles",
        ":OpBaseTdFiles",
        ":StdOpsTdFiles",
        ":VectorOpsTdFiles",
    ],
)

gentbl(
    name = "VectorTransformPatternsIncGen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "include/mlir/Dialect/VectorOps/VectorTransformPatterns.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/VectorOps/VectorTransformPatterns.td",
    td_srcs = [
        ":VectorTransformPatternsTdFiles",
    ],
)

cc_library(
    name = "VectorToLLVM",
    srcs = glob([
        "lib/Conversion/VectorToLLVM/*.cpp",
        "lib/Conversion/VectorToLLVM/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Conversion/VectorToLLVM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":EDSC",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
        ":VectorOps",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "VectorToLoops",
    srcs = glob([
        "lib/Conversion/VectorToLoops/*.cpp",
        "lib/Conversion/VectorToLoops/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Conversion/VectorToLoops/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":EDSC",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
        ":VectorOps",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

gentbl(
    name = "tpu_intrinsics_patterns_target",
    strip_include_prefix = "include",
    tbl_outs = [("-gen-llvmir-intrinsics -dialect-opclass-base LLVM_TpuOp -llvmir-intrinsics-filter int_tpu", "include/mlir/IR/llvm_tpu_ops_gen.td.inc")],
    tblgen = ":mlir-tblgen",
    td_file = "//third_party/llvm/llvm-project/llvm:include/llvm/IR/Intrinsics.td",
    td_includes = ["third_party/llvm/llvm-project/llvm/include/"],
    td_srcs = ["//third_party/llvm/llvm-project/llvm:common_target_td_sources"],
)

# To reference all tablegen files here when checking for updates to them.
filegroup(
    name = "TdFiles",
    srcs = glob(["**/*.td"]),
)

exports_files(
    [
        "include/mlir/Analysis/CallInterfaces.h",
        "include/mlir/Analysis/CallInterfaces.td",
        "include/mlir/Dialect/LLVMIR/LLVMOpBase.td",
        "include/mlir/Dialect/StandardOps/Ops.td",
        "include/mlir/IR/OpAsmInterface.td",
        "include/mlir/IR/OpBase.td",
        "include/mlir/Transforms/InliningUtils.h",
    ],
    visibility = ["@llvm-project//mlir:friends"],
)

exports_files(
    [
        "include/mlir/Analysis/InferTypeOpInterface.td",
        "include/mlir/Transforms/LoopLikeInterface.td",
    ],
    visibility = ["@llvm-project//mlir:friends"],
)
