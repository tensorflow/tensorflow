# Description:
#   The MLIR "Multi-Level Intermediate Representation" Compiler Infrastructure

load(":tblgen.bzl", "gentbl")

licenses(["notice"])

package(default_visibility = [":friends"])

package_group(
    name = "subpackages",
    packages = ["//..."],
)

# In particular the OWNERS file of the dependent project should be updated.
# TODO(b/140669524): Use proper MLIR tests instead of end-to-end tests for
# tf_runtime and tf_runtime_google.
package_group(
    name = "friends",
    includes = ["@org_tensorflow//tensorflow/compiler/mlir:subpackages"],
    packages = [
        "//...",
        "//learning/brain/research/sair/...",
        "//learning/brain/swift/swift_mlir/...",
        "//learning/glassbox/evaluation/compiler/...",
        "//tensorflow/compiler/mlir/tfrt/...",
        "//tensorflow/core/tfrt_delegate/...",
        "//tensorflow/lite/experimental/tf_runtime/...",
        "//third_party/tf_runtime_google/...",
    ],
)

exports_files([
    "run_lit.sh",
    "LICENSE.TXT",
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
    srcs = [
        "lib/IR/AffineExpr.cpp",
        "lib/IR/AffineExprDetail.h",
        "lib/IR/AffineMap.cpp",
        "lib/IR/AffineMapDetail.h",
        "lib/IR/AsmPrinter.cpp",
        "lib/IR/AttributeDetail.h",
        "lib/IR/Attributes.cpp",
        "lib/IR/Block.cpp",
        "lib/IR/Builders.cpp",
        "lib/IR/Diagnostics.cpp",
        "lib/IR/Dialect.cpp",
        "lib/IR/Function.cpp",
        "lib/IR/FunctionImplementation.cpp",
        "lib/IR/IntegerSet.cpp",
        "lib/IR/IntegerSetDetail.h",
        "lib/IR/Location.cpp",
        "lib/IR/LocationDetail.h",
        "lib/IR/MLIRContext.cpp",
        "lib/IR/Module.cpp",
        "lib/IR/Operation.cpp",
        "lib/IR/OperationSupport.cpp",
        "lib/IR/PatternMatch.cpp",
        "lib/IR/Region.cpp",
        "lib/IR/StandardTypes.cpp",
        "lib/IR/SymbolTable.cpp",
        "lib/IR/TypeDetail.h",
        "lib/IR/TypeUtilities.cpp",
        "lib/IR/Types.cpp",
        "lib/IR/Value.cpp",
        "lib/IR/Visitors.cpp",
    ],
    hdrs = [
        "include/mlir/Analysis/CallInterfaces.h",
        "include/mlir/IR/AffineExpr.h",
        "include/mlir/IR/AffineExprVisitor.h",
        "include/mlir/IR/AffineMap.h",
        "include/mlir/IR/AttributeSupport.h",
        "include/mlir/IR/Attributes.h",
        "include/mlir/IR/Block.h",
        "include/mlir/IR/BlockAndValueMapping.h",
        "include/mlir/IR/BlockSupport.h",
        "include/mlir/IR/Builders.h",
        "include/mlir/IR/Diagnostics.h",
        "include/mlir/IR/Dialect.h",
        "include/mlir/IR/DialectHooks.h",
        "include/mlir/IR/DialectImplementation.h",
        "include/mlir/IR/DialectInterface.h",
        "include/mlir/IR/Function.h",
        "include/mlir/IR/FunctionImplementation.h",
        "include/mlir/IR/FunctionSupport.h",
        "include/mlir/IR/Identifier.h",
        "include/mlir/IR/IntegerSet.h",
        "include/mlir/IR/Location.h",
        "include/mlir/IR/MLIRContext.h",
        "include/mlir/IR/Matchers.h",
        "include/mlir/IR/Module.h",
        "include/mlir/IR/OpDefinition.h",
        "include/mlir/IR/OpImplementation.h",
        "include/mlir/IR/Operation.h",
        "include/mlir/IR/OperationSupport.h",
        "include/mlir/IR/PatternMatch.h",
        "include/mlir/IR/Region.h",
        "include/mlir/IR/RegionGraphTraits.h",
        "include/mlir/IR/StandardTypes.h",
        "include/mlir/IR/StorageUniquerSupport.h",
        "include/mlir/IR/SymbolTable.h",
        "include/mlir/IR/TypeSupport.h",
        "include/mlir/IR/TypeUtilities.h",
        "include/mlir/IR/Types.h",
        "include/mlir/IR/UseDefLists.h",
        "include/mlir/IR/Value.h",
        "include/mlir/IR/Visitors.h",
    ],
    includes = ["include"],
    deps = [
        ":CallOpInterfacesIncGen",
        ":DialectSymbolRegistry",
        ":InferTypeOpInterfaceIncGen",
        ":OpAsmInterfacesIncGen",
        ":Support",
        "@llvm//:support",
    ],
)

cc_library(
    name = "Pass",
    srcs = [
        "lib/Pass/IRPrinting.cpp",
        "lib/Pass/Pass.cpp",
        "lib/Pass/PassDetail.h",
        "lib/Pass/PassManagerOptions.cpp",
        "lib/Pass/PassRegistry.cpp",
        "lib/Pass/PassStatistics.cpp",
        "lib/Pass/PassTiming.cpp",
    ],
    hdrs = [
        "include/mlir/Analysis/Verifier.h",
        "include/mlir/Pass/AnalysisManager.h",
        "include/mlir/Pass/Pass.h",
        "include/mlir/Pass/PassInstrumentation.h",
        "include/mlir/Pass/PassManager.h",
        "include/mlir/Pass/PassRegistry.h",
    ],
    includes = ["include"],
    linkopts = [
        "-lm",
        "-lpthread",
    ],
    deps = [
        ":IR",
        ":Support",
        "@llvm//:support",
    ],
)

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
        "@llvm//:support",
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
        "@llvm//:support",
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
    srcs = [
        "lib/Dialect/Traits.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/Traits.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        "@llvm//:support",
    ],
)

cc_library(
    name = "DialectUtils",
    srcs = [
    ],
    hdrs = [
        "include/mlir/Dialect/Utils/StructuredOpsUtils.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":Support",
        "@llvm//:support",
    ],
)

cc_library(
    name = "AffineOps",
    srcs = [
        "include/mlir/Transforms/InliningUtils.h",
        "include/mlir/Transforms/LoopLikeInterface.h",
        "lib/Dialect/AffineOps/AffineOps.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/AffineOps/AffineOps.h",
        "include/mlir/Transforms/SideEffectsInterface.h",
    ],
    includes = ["include"],
    deps = [
        ":AffineOpsIncGen",
        ":IR",
        ":LoopLikeOpInterfaceIncGen",
        ":StandardOps",
        ":Support",
        "@llvm//:support",
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
    srcs = ["lib/Conversion/AffineToStandard/AffineToStandard.cpp"],
    hdrs = ["include/mlir/Conversion/AffineToStandard/AffineToStandard.h"],
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
    srcs = [
        "lib/Dialect/SDBM/SDBM.cpp",
        "lib/Dialect/SDBM/SDBMDialect.cpp",
        "lib/Dialect/SDBM/SDBMExpr.cpp",
        "lib/Dialect/SDBM/SDBMExprDetail.h",
    ],
    hdrs = [
        "include/mlir/Dialect/SDBM/SDBM.h",
        "include/mlir/Dialect/SDBM/SDBMDialect.h",
        "include/mlir/Dialect/SDBM/SDBMExpr.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":Support",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "LoopOps",
    srcs = [
        "lib/Dialect/LoopOps/LoopOps.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/LoopOps/LoopOps.h",
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
        "@llvm//:support",
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
    srcs = [
        "lib/Dialect/StandardOps/Ops.cpp",
    ],
    hdrs = [
        "include/mlir/Analysis/CallInterfaces.h",
        "include/mlir/Dialect/StandardOps/Ops.h",
        "include/mlir/Transforms/InliningUtils.h",
    ],
    includes = ["include"],
    deps = [
        ":CallOpInterfacesIncGen",
        ":CommonFolders",
        ":IR",
        ":StandardOpsIncGen",
        ":Support",
        "@llvm//:support",
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
    srcs = [
        "lib/Dialect/VectorOps/VectorOps.cpp",
        "lib/Dialect/VectorOps/VectorTransforms.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/VectorOps/Utils.h",
        "include/mlir/Dialect/VectorOps/VectorOps.h",
        "include/mlir/Dialect/VectorOps/VectorTransforms.h",
    ],
    includes = ["include"],
    deps = [
        ":DialectUtils",
        ":EDSC",
        ":IR",
        ":StandardOps",
        ":Support",
        ":VectorOpsIncGen",
        ":VectorTransformPatternsIncGen",
        "@llvm//:support",
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
    srcs = [
        "lib/Support/FileUtilities.cpp",
        "lib/Support/StorageUniquer.cpp",
        "lib/Support/ToolUtilities.cpp",
    ],
    hdrs = [
        "include/mlir/Support/DebugStringHelper.h",
        "include/mlir/Support/FileUtilities.h",
        "include/mlir/Support/Functional.h",
        "include/mlir/Support/LLVM.h",
        "include/mlir/Support/LogicalResult.h",
        "include/mlir/Support/MathExtras.h",
        "include/mlir/Support/STLExtras.h",
        "include/mlir/Support/StorageUniquer.h",
        "include/mlir/Support/StringExtras.h",
        "include/mlir/Support/ToolUtilities.h",
    ],
    includes = ["include"],
    deps = [
        "@llvm//:support",
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
    srcs = [
        "lib/Parser/Lexer.cpp",
        "lib/Parser/Lexer.h",
        "lib/Parser/Parser.cpp",
        "lib/Parser/Token.cpp",
        "lib/Parser/Token.h",
    ],
    hdrs = [
        "include/mlir/Parser.h",
    ],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":IR",
        ":ParserTokenKinds",
        ":Support",
        "@llvm//:support",
    ],
)

cc_library(
    name = "LLVMDialect",
    srcs = [
        "lib/Dialect/LLVMIR/IR/LLVMDialect.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/LLVMIR/LLVMDialect.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMOpsIncGen",
        ":Support",
        "@llvm//:asm_parser",
        "@llvm//:core",
        "@llvm//:support",
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
    srcs = ["lib/Dialect/GPU/IR/GPUDialect.cpp"],
    hdrs = [
        "include/mlir/Dialect/GPU/GPUDialect.h",
    ],
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
    srcs = ["lib/Dialect/GPU/Transforms/KernelOutlining.cpp"],
    hdrs = ["include/mlir/Dialect/GPU/Passes.h"],
    includes = ["include"],
    deps = [
        ":GPUDialect",
        ":IR",
        ":Pass",
        ":StandardOps",
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
        "@llvm//:support",
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
    srcs = ["lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp"],
    hdrs = [
        "include/mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h",
    ],
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
        "@llvm//:support",
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
        "@llvm//:core",
        "@llvm//:nvptx_target",  # buildcleaner: keep
        "@llvm//:support",
        "@llvm//:target",
    ],
    alwayslink = 1,
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
    includes = ["include"],
    deps = [
        ":GPUDialect",
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
        "@llvm//:asm_parser",
        "@llvm//:core",
        "@llvm//:support",
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
        "@llvm//:asm_parser",
        "@llvm//:core",
        "@llvm//:support",
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

filegroup(
    name = "SPIRVOpsTdFiles",
    srcs = [
        "include/mlir/Analysis/CallInterfaces.td",
        "include/mlir/Dialect/SPIRV/SPIRVArithmeticOps.td",
        "include/mlir/Dialect/SPIRV/SPIRVAtomicOps.td",
        "include/mlir/Dialect/SPIRV/SPIRVBase.td",
        "include/mlir/Dialect/SPIRV/SPIRVBitOps.td",
        "include/mlir/Dialect/SPIRV/SPIRVCastOps.td",
        "include/mlir/Dialect/SPIRV/SPIRVCompositeOps.td",
        "include/mlir/Dialect/SPIRV/SPIRVControlFlowOps.td",
        "include/mlir/Dialect/SPIRV/SPIRVGLSLOps.td",
        "include/mlir/Dialect/SPIRV/SPIRVGroupOps.td",
        "include/mlir/Dialect/SPIRV/SPIRVLogicalOps.td",
        "include/mlir/Dialect/SPIRV/SPIRVNonUniformOps.td",
        "include/mlir/Dialect/SPIRV/SPIRVOps.td",
        "include/mlir/Dialect/SPIRV/SPIRVStructureOps.td",
        ":OpBaseTdFiles",
    ],
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
    name = "SPIRVLoweringStructGen",
    tbl_outs = [
        (
            "-gen-struct-attr-decls",
            "include/mlir/Dialect/SPIRV/SPIRVLowering.h.inc",
        ),
        (
            "-gen-struct-attr-defs",
            "include/mlir/Dialect/SPIRV/SPIRVLowering.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/SPIRVLowering.td",
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
    srcs = [
        "include/mlir/Transforms/InliningUtils.h",
        "lib/Dialect/SPIRV/LayoutUtils.cpp",
        "lib/Dialect/SPIRV/SPIRVDialect.cpp",
        "lib/Dialect/SPIRV/SPIRVOps.cpp",
        "lib/Dialect/SPIRV/SPIRVTypes.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/SPIRV/LayoutUtils.h",
        "include/mlir/Dialect/SPIRV/SPIRVDialect.h",
        "include/mlir/Dialect/SPIRV/SPIRVOps.h",
        "include/mlir/Dialect/SPIRV/SPIRVTypes.h",
    ],
    includes = ["include"],
    deps = [
        ":CommonFolders",
        ":IR",
        ":Parser",
        ":SPIRVCanonicalizationIncGen",
        ":SPIRVOpUtilsIncGen",
        ":SPIRVOpsIncGen",
        ":Support",
        "@llvm//:support",
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
    ],
    includes = [
        "include",
    ],
    deps = [
        ":IR",
        ":Pass",
        ":SPIRVDialect",
        ":SPIRVLoweringStructGen",
        ":StandardOps",
        ":Support",
        ":Transforms",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "StandardToSPIRVConversions",
    srcs = [
        "lib/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.cpp",
        "lib/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.cpp",
        "lib/Conversion/StandardToSPIRV/LegalizeStandardForSPIRV.cpp",
    ],
    hdrs = [
        "include/mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h",
        "include/mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h",
    ],
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
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "SPIRVSerialization",
    srcs = [
        "lib/Dialect/SPIRV/Serialization/Deserializer.cpp",
        "lib/Dialect/SPIRV/Serialization/SPIRVBinaryUtils.cpp",
        "lib/Dialect/SPIRV/Serialization/Serializer.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/SPIRV/SPIRVBinaryUtils.h",
        "include/mlir/Dialect/SPIRV/Serialization.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":SPIRVDialect",
        ":SPIRVSerializationGen",
        ":Support",
        "@llvm//:support",
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
        "@llvm//:support",
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
    srcs = [
        "lib/Transforms/Utils/FoldUtils.cpp",
        "lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp",
        "lib/Transforms/Utils/InliningUtils.cpp",
        "lib/Transforms/Utils/LoopFusionUtils.cpp",
        "lib/Transforms/Utils/LoopUtils.cpp",
        "lib/Transforms/Utils/RegionUtils.cpp",
        "lib/Transforms/Utils/Utils.cpp",
    ],
    hdrs = [
        "include/mlir/Transforms/FoldUtils.h",
        "include/mlir/Transforms/InliningUtils.h",
        "include/mlir/Transforms/LoopFusionUtils.h",
        "include/mlir/Transforms/LoopUtils.h",
        "include/mlir/Transforms/RegionUtils.h",
        "include/mlir/Transforms/Utils.h",
    ],
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":Analysis",
        ":IR",
        ":LoopDialectRegistration",
        ":LoopOps",
        ":StandardDialectRegistration",
        ":StandardOps",
        ":Support",
        "@llvm//:support",
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
    srcs = [
        "lib/Transforms/AffineDataCopyGeneration.cpp",
        "lib/Transforms/AffineLoopInvariantCodeMotion.cpp",
        "lib/Transforms/CSE.cpp",
        "lib/Transforms/Canonicalizer.cpp",
        "lib/Transforms/DialectConversion.cpp",
        "lib/Transforms/Inliner.cpp",
        "lib/Transforms/LoopCoalescing.cpp",
        "lib/Transforms/LoopFusion.cpp",
        "lib/Transforms/LoopInvariantCodeMotion.cpp",
        "lib/Transforms/LoopTiling.cpp",
        "lib/Transforms/LoopUnroll.cpp",
        "lib/Transforms/LoopUnrollAndJam.cpp",
        "lib/Transforms/MemRefDataFlowOpt.cpp",
        "lib/Transforms/PipelineDataTransfer.cpp",
        "lib/Transforms/SimplifyAffineStructures.cpp",
        "lib/Transforms/StripDebugInfo.cpp",
        "lib/Transforms/Vectorize.cpp",
    ],
    hdrs = [
        "include/mlir/Transforms/DialectConversion.h",
        "include/mlir/Transforms/Passes.h",
        "include/mlir/Transforms/SideEffectsInterface.h",
    ],
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
        "@llvm//:support",
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
        "@llvm//:support",
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
        ":Linalg",
        ":LoopOps",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":Transforms",
        "@llvm//:support",
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
        "@llvm//:support",
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
        "@llvm//:core",
        "@llvm//:support",
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
    srcs = [
        "lib/Analysis/AffineAnalysis.cpp",
        "lib/Analysis/AffineStructures.cpp",
        "lib/Analysis/CallGraph.cpp",
        "lib/Analysis/Dominance.cpp",
        "lib/Analysis/InferTypeOpInterface.cpp",
        "lib/Analysis/Liveness.cpp",
        "lib/Analysis/LoopAnalysis.cpp",
        "lib/Analysis/MemRefBoundCheck.cpp",
        "lib/Analysis/NestedMatcher.cpp",
        "lib/Analysis/OpStats.cpp",
        "lib/Analysis/SliceAnalysis.cpp",
        "lib/Analysis/TestMemRefDependenceCheck.cpp",
        "lib/Analysis/TestParallelismDetection.cpp",
        "lib/Analysis/Utils.cpp",
        "lib/Analysis/Verifier.cpp",
    ],
    hdrs = [
        "include/mlir/Analysis/AffineAnalysis.h",
        "include/mlir/Analysis/AffineStructures.h",
        "include/mlir/Analysis/CallGraph.h",
        "include/mlir/Analysis/CallInterfaces.h",
        "include/mlir/Analysis/Dominance.h",
        "include/mlir/Analysis/InferTypeOpInterface.h",
        "include/mlir/Analysis/Liveness.h",
        "include/mlir/Analysis/LoopAnalysis.h",
        "include/mlir/Analysis/NestedMatcher.h",
        "include/mlir/Analysis/Passes.h",
        "include/mlir/Analysis/SliceAnalysis.h",
        "include/mlir/Analysis/Utils.h",
        "include/mlir/Analysis/Verifier.h",
    ],
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
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "VectorAnalysis",
    srcs = [
        "lib/Analysis/VectorAnalysis.cpp",
    ],
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":Analysis",
        ":IR",
        ":StandardOps",
        ":Support",
        ":VectorOps",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "Translation",
    srcs = ["lib/Translation/Translation.cpp"],
    hdrs = ["include/mlir/Translation.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":Parser",
        ":StandardOps",
        ":Support",
        "@llvm//:support",
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
        "@llvm//:core",
        "@llvm//:support",
        "@llvm//:transform_utils",
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
        "@llvm//:core",
        "@llvm//:ir_reader",
        "@llvm//:support",
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
        "@llvm//:core",
        "@llvm//:support",
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
        "@llvm//:core",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

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
        "@llvm//:bit_reader",
        "@llvm//:bit_writer",
        "@llvm//:core",
        "@llvm//:execution_engine",
        "@llvm//:mc",
        "@llvm//:orc_jit",
        "@llvm//:support",
        "@llvm//:target",  # fixdeps: keep
        "@llvm//:transform_utils",
        "@llvm//:x86_code_gen",  # fixdeps: keep
        "@llvm//:x86_disassembler",  # fixdeps: keep
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
        "@llvm//:analysis",
        "@llvm//:core",
        "@llvm//:ipo",
        "@llvm//:support",
        "@llvm//:target",
    ],
)

cc_library(
    name = "MlirOptLib",
    srcs = [
        "lib/Support/MlirOptMain.cpp",
    ],
    hdrs = [
        "include/mlir/Support/MlirOptMain.h",
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
        ":ViewOpGraph",
        ":ViewRegionGraph",
        "@llvm//:support",
    ],
)

cc_library(
    name = "ViewOpGraph",
    srcs = ["lib/Transforms/ViewOpGraph.cpp"],
    hdrs = ["include/mlir/Transforms/ViewOpGraph.h"],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":IR",
        ":Pass",
        ":Support",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ViewRegionGraph",
    srcs = ["lib/Transforms/ViewRegionGraph.cpp"],
    hdrs = ["include/mlir/Transforms/ViewRegionGraph.h"],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":IR",
        ":Pass",
        ":Support",
        "@llvm//:support",
    ],
    alwayslink = 1,
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
        "@llvm//:support",
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
        "@llvm//:support",
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
        "@llvm//:support",
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
        ":QuantOps",
        ":QuantOpsDialectRegistration",
        ":ROCDLDialect",
        ":StandardDialectRegistration",
        ":Transforms",
        ":VectorDialectRegistration",
        "//test:TestDialect",
        "//test:TestTransforms",
        "@llvm//:support",
        "@local_config_mlir//test:TestIR",
        "@local_config_mlir//test:TestPass",
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
        "@llvm//:core",
        "@llvm//:orc_jit",
        "@llvm//:support",
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
        "@llvm//:support",
    ],
)

cc_binary(
    name = "mlir-cuda-runner",
    srcs = ["tools/mlir-cuda-runner/mlir-cuda-runner.cpp"],
    data = [
        ":tools/libcuda-runtime-wrappers.so",
        "@local_config_mlir//test/mlir-cpu-runner:libmlir_runner_utils.so",
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
        "@llvm//:support",
    ],
)

cc_library(
    name = "TableGen",
    srcs = [
        "lib/TableGen/Argument.cpp",
        "lib/TableGen/Attribute.cpp",
        "lib/TableGen/Constraint.cpp",
        "lib/TableGen/Dialect.cpp",
        "lib/TableGen/Format.cpp",
        "lib/TableGen/OpInterfaces.cpp",
        "lib/TableGen/OpTrait.cpp",
        "lib/TableGen/Operator.cpp",
        "lib/TableGen/Pattern.cpp",
        "lib/TableGen/Predicate.cpp",
        "lib/TableGen/Type.cpp",
    ],
    hdrs = [
        "include/mlir/TableGen/Argument.h",
        "include/mlir/TableGen/Attribute.h",
        "include/mlir/TableGen/Constraint.h",
        "include/mlir/TableGen/Dialect.h",
        "include/mlir/TableGen/Format.h",
        "include/mlir/TableGen/GenInfo.h",
        "include/mlir/TableGen/GenNameParser.h",
        "include/mlir/TableGen/OpInterfaces.h",
        "include/mlir/TableGen/OpTrait.h",
        "include/mlir/TableGen/Operator.h",
        "include/mlir/TableGen/Pattern.h",
        "include/mlir/TableGen/Predicate.h",
        "include/mlir/TableGen/Region.h",
        "include/mlir/TableGen/Type.h",
    ],
    includes = ["include"],
    deps = [
        ":Support",
        "@llvm//:support",
        "@llvm//:tablegen",
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
        "@llvm//:config",
        "@llvm//:support",
        "@llvm//:tablegen",
    ],
)

cc_binary(
    name = "mlir-tblgen",
    srcs = [
        "tools/mlir-tblgen/DocGenUtilities.h",
        "tools/mlir-tblgen/EnumsGen.cpp",
        "tools/mlir-tblgen/LLVMIRConversionGen.cpp",
        "tools/mlir-tblgen/OpDefinitionsGen.cpp",
        "tools/mlir-tblgen/OpDocGen.cpp",
        "tools/mlir-tblgen/OpInterfacesGen.cpp",
        "tools/mlir-tblgen/ReferenceImplGen.cpp",
        "tools/mlir-tblgen/RewriterGen.cpp",
        "tools/mlir-tblgen/SPIRVUtilsGen.cpp",
        "tools/mlir-tblgen/StructsGen.cpp",
    ],
    linkopts = [
        "-lm",
        "-lpthread",
    ],
    deps = [
        ":MlirTableGenMain",
        ":Support",
        ":TableGen",
        "@llvm//:config",
        "@llvm//:support",
        "@llvm//:tablegen",
    ],
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
        "@llvm//:support",
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
        "@llvm//:support",
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
    name = "LinalgLibraryOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Linalg/IR/LinalgBase.td",
        "include/mlir/Dialect/Linalg/IR/LinalgLibraryOps.td",
        ":AffineOpsTdFiles",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "LinalgLibraryOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/Linalg/IR/LinalgLibraryOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/Linalg/IR/LinalgLibraryOps.cpp.inc",
        ),
        (
            "-gen-op-interface-decls",
            "include/mlir/Dialect/Linalg/IR/LinalgLibraryOpInterfaces.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "include/mlir/Dialect/Linalg/IR/LinalgLibraryOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/IR/LinalgLibraryOps.td",
    td_srcs = [
        ":LinalgLibraryOpsTdFiles",
    ],
)

filegroup(
    name = "LinalgTransformPatternsTdFiles",
    srcs = [
        "include/mlir/Dialect/Linalg/Transforms/LinalgTransformPatterns.td",
        ":AffineOpsTdFiles",
        ":LinalgLibraryOpsTdFiles",
        ":LinalgOpsTdFiles",
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
    srcs = [
        "lib/Conversion/LinalgToLLVM/LinalgToLLVM.cpp",
    ],
    hdrs = [
        "include/mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h",
    ],
    includes = ["include"],
    deps = [
        ":AffineToStandardTransforms",
        ":Analysis",
        ":CFGTransforms",
        ":EDSC",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":Linalg",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
        ":VectorToLLVM",
        "@llvm//:core",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "Linalg",
    srcs = [
        "lib/Dialect/Linalg/Analysis/DependenceAnalysis.cpp",
        "lib/Dialect/Linalg/IR/LinalgOps.cpp",
        "lib/Dialect/Linalg/IR/LinalgTypes.cpp",
        "lib/Dialect/Linalg/Transforms/Fusion.cpp",
        "lib/Dialect/Linalg/Transforms/LinalgToLoops.cpp",
        "lib/Dialect/Linalg/Transforms/LinalgTransforms.cpp",
        "lib/Dialect/Linalg/Transforms/Promotion.cpp",
        "lib/Dialect/Linalg/Transforms/Tiling.cpp",
        "lib/Dialect/Linalg/Utils/Utils.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h",
        "include/mlir/Dialect/Linalg/IR/LinalgOps.h",
        "include/mlir/Dialect/Linalg/IR/LinalgTraits.h",
        "include/mlir/Dialect/Linalg/IR/LinalgTypes.h",
        "include/mlir/Dialect/Linalg/Passes.h",
        "include/mlir/Dialect/Linalg/Transforms/LinalgTransforms.h",
        "include/mlir/Dialect/Linalg/Utils/Intrinsics.h",
        "include/mlir/Dialect/Linalg/Utils/Utils.h",
    ],
    includes = ["include"],
    deps = [
        "LinalgTransformPatternsIncGen",
        ":AffineOps",
        ":AffineToStandardTransforms",
        ":Analysis",
        ":CFGTransforms",
        ":EDSC",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":LinalgLibraryOpsIncGen",
        ":LinalgOpsIncGen",
        ":LoopOps",
        ":Parser",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":Transforms",
        ":VectorOps",
        "@llvm//:core",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "LinalgDialectRegistration",
    srcs = ["lib/Dialect/Linalg/LinalgRegistration.cpp"],
    deps = [":Linalg"],
    alwayslink = 1,
)

cc_library(
    name = "QuantizerSupportLib",
    srcs = [
        "lib/Quantizer/Configurations/FxpMathConfig.cpp",
        "lib/Quantizer/Support/Configuration.cpp",
        "lib/Quantizer/Support/ConstraintAnalysisGraph.cpp",
        "lib/Quantizer/Support/Metadata.cpp",
        "lib/Quantizer/Support/Statistics.cpp",
        "lib/Quantizer/Support/TypeUtils.cpp",
        "lib/Quantizer/Support/UniformConstraints.cpp",
        "lib/Quantizer/Support/UniformSolvers.cpp",
    ],
    hdrs = [
        "include/mlir/Quantizer/Configurations/FxpMathConfig.h",
        "include/mlir/Quantizer/Support/Configuration.h",
        "include/mlir/Quantizer/Support/ConstraintAnalysisGraph.h",
        "include/mlir/Quantizer/Support/ConstraintAnalysisGraphTraits.h",
        "include/mlir/Quantizer/Support/Metadata.h",
        "include/mlir/Quantizer/Support/Rules.h",
        "include/mlir/Quantizer/Support/Statistics.h",
        "include/mlir/Quantizer/Support/TypeUtils.h",
        "include/mlir/Quantizer/Support/UniformConstraints.h",
        "include/mlir/Quantizer/Support/UniformSolvers.h",
    ],
    includes = ["include"],
    deps = [
        ":FxpMathOps",
        ":IR",
        ":QuantOps",
        ":StandardOps",
        "@llvm//:support",
    ],
)

cc_library(
    name = "QuantizerTransforms",
    srcs = [
        "lib/Quantizer/Transforms/AddDefaultStatsTestPass.cpp",
        "lib/Quantizer/Transforms/InferQuantizedTypesPass.cpp",
        "lib/Quantizer/Transforms/RemoveInstrumentationPass.cpp",
    ],
    hdrs = [
        "include/mlir/Quantizer/Transforms/Passes.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":Pass",
        ":QuantOps",
        ":QuantizerSupportLib",
        ":Support",
        "@llvm//:support",
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
        ":LinalgLibraryOpsTdFiles",
        ":LinalgOpsTdFiles",
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
    srcs = [
        "lib/Conversion/VectorToLLVM/ConvertVectorToLLVM.cpp",
    ],
    hdrs = [
        "include/mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h",
    ],
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
        "@llvm//:core",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "VectorToLoops",
    srcs = [
        "lib/Conversion/VectorToLoops/ConvertVectorToLoops.cpp",
    ],
    hdrs = [
        "include/mlir/Conversion/VectorToLoops/ConvertVectorToLoops.h",
    ],
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
        "@llvm//:core",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

# To reference all tablegen files here when checking for updates to them.
filegroup(
    name = "TdFiles",
    srcs = glob(["**/*.td"]),
)

exports_files(
    [
        "include/mlir/Dialect/StandardOps/Ops.td",
        "include/mlir/Analysis/CallInterfaces.td",
        "include/mlir/Transforms/InliningUtils.h",
        "include/mlir/IR/OpBase.td",
        "include/mlir/IR/OpAsmInterface.td",
        "include/mlir/Analysis/CallInterfaces.h",
    ],
    visibility = ["@local_config_mlir//:friends"],
)

exports_files(
    ["include/mlir/Analysis/InferTypeOpInterface.td"],
    visibility = ["@local_config_mlir//:friends"],
)
