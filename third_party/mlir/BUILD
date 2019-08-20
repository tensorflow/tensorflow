# Description:
#   The MLIR "Multi-Level Intermediate Representation" Compiler Infrastructure

licenses(["notice"])

package(
    default_visibility = [":friends"],
    features = ["-layering_check"],
)

package_group(
    name = "subpackages",
    packages = ["//..."],
)

# Before adding a project here, please read go/mlir-sla
# In particular the OWNERS file of the dependent project should be updated.
package_group(
    name = "friends",
    includes = ["@org_tensorflow//tensorflow/compiler/mlir:subpackages"],
    packages = [
        "//...",
        "//learning/glassbox/evaluation/compiler/...",
        "//tensorflow/compiler/xla/service/gpu/mlir/...",
    ],
)

exports_files([
    "run_lit.sh",
    "LICENSE.TXT",
])

load(":tblgen.bzl", "gentbl")

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
        "lib/IR/FunctionSupport.cpp",
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
    ],
    hdrs = [
        "include/mlir/IR/AffineExpr.h",
        "include/mlir/IR/AffineExprVisitor.h",
        "include/mlir/IR/AffineMap.h",
        "include/mlir/IR/AttributeSupport.h",
        "include/mlir/IR/Attributes.h",
        "include/mlir/IR/Block.h",
        "include/mlir/IR/BlockAndValueMapping.h",
        "include/mlir/IR/Builders.h",
        "include/mlir/IR/Diagnostics.h",
        "include/mlir/IR/Dialect.h",
        "include/mlir/IR/DialectHooks.h",
        "include/mlir/IR/DialectInterface.h",
        "include/mlir/IR/DialectSymbolRegistry.def",
        "include/mlir/IR/Function.h",
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
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
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
    copts = ["-std=c++14"],
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
        "lib/EDSC/CoreAPIs.cpp",
        "lib/EDSC/Helpers.cpp",
        "lib/EDSC/Intrinsics.cpp",
    ],
    hdrs = [
        "include/mlir-c/Core.h",
        "include/mlir/EDSC/Builders.h",
        "include/mlir/EDSC/Helpers.h",
        "include/mlir/EDSC/Intrinsics.h",
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":Analysis",
        ":IR",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":VectorOps",
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
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "AffineOpsIncGen",
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
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "LoopOpsIncGen",
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
        "include/mlir/Dialect/StandardOps/Ops.td",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "StandardOpsIncGen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/StandardOps/Ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/StandardOps/Ops.cpp.inc",
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
    copts = ["-std=c++14"],
    deps = [
        ":IR",
        "@llvm//:support",
    ],
)

cc_library(
    name = "AffineOps",
    srcs = [
        "lib/Dialect/AffineOps/AffineOps.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/AffineOps/AffineOps.h",
    ],
    copts = ["-std=c++14"],
    deps = [
        ":AffineOpsIncGen",
        ":IR",
        ":StandardOps",
        ":Support",
        "@llvm//:support",
    ],
)

# Library with affine dialect static initialization.
cc_library(
    name = "AffineDialectRegistration",
    srcs = ["lib/Dialect/AffineOps/DialectRegistration.cpp"],
    copts = ["-std=c++14"],
    deps = [
        ":AffineOps",
        ":IR",
    ],
    alwayslink = 1,
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
    copts = ["-std=c++14"],
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
    ],
    copts = ["-std=c++14"],
    deps = [
        ":IR",
        ":LoopOpsIncGen",
        ":StandardOps",
        ":Support",
        "@llvm//:support",
    ],
)

cc_library(
    name = "LoopDialectRegistration",
    srcs = ["lib/Dialect/LoopOps/DialectRegistration.cpp"],
    copts = ["-std=c++14"],
    deps = [
        ":IR",
        ":LoopOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "StandardOps",
    srcs = [
        "lib/Dialect/StandardOps/Ops.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/StandardOps/Ops.h",
    ],
    copts = ["-std=c++14"],
    deps = [
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
    copts = ["-std=c++14"],
    deps = [
        ":IR",
        ":StandardOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "VectorOps",
    srcs = [
        "lib/Dialect/VectorOps/VectorOps.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/VectorOps/VectorOps.h",
    ],
    copts = ["-std=c++14"],
    deps = [
        ":IR",
        ":Support",
        ":VectorOpsIncGen",
        "@llvm//:support",
    ],
)

cc_library(
    name = "VectorDialectRegistration",
    srcs = ["lib/Dialect/VectorOps/DialectRegistration.cpp"],
    copts = ["-std=c++14"],
    deps = [
        ":IR",
        ":VectorOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "Support",
    srcs = [
        "lib/Support/FileUtilities.cpp",
        "lib/Support/StorageUniquer.cpp",
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
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        "@llvm//:support",
    ],
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
        "lib/Parser/TokenKinds.def",
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":IR",
        ":Support",
        "@llvm//:support",
    ],
)

cc_library(
    name = "LLVMDialect",
    srcs = [
        "include/mlir/Dialect/LLVMIR/LLVMOps.cpp.inc",
        "include/mlir/Dialect/LLVMIR/LLVMOps.h.inc",
        "lib/Dialect/LLVMIR/IR/LLVMDialect.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/LLVMIR/LLVMDialect.h",
    ],
    copts = ["-std=c++14"],
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
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "GPUOpsIncGen",
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
    hdrs = ["include/mlir/Dialect/GPU/GPUDialect.h"],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":GPUOpsIncGen",
        ":IR",
        ":StandardOps",
        ":Support",
    ],
)

cc_library(
    name = "GPUDialectRegistration",
    srcs = ["lib/Dialect/GPU/IR/DialectRegistration.cpp"],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":GPUDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "GPUTransforms",
    srcs = ["lib/Dialect/GPU/Transforms/KernelOutlining.cpp"],
    hdrs = ["include/mlir/Dialect/GPU/Passes.h"],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":GPUDialect",
        ":IR",
        ":LoopOps",
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
    name = "GPUToNVVMTransforms",
    srcs = ["lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp"],
    hdrs = [
        "include/mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h",
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":GPUDialect",
        ":IR",
        ":LLVMDialect",
        ":NVVMDialect",
        ":Pass",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "GPUToCUDATransforms",
    srcs = [
        "lib/Conversion/GPUToCUDA/ConvertKernelFuncToCubin.cpp",
        "lib/Conversion/GPUToCUDA/ConvertLaunchFuncToCudaCalls.cpp",
        "lib/Conversion/GPUToCUDA/GenerateCubinAccessors.cpp",
    ],
    hdrs = ["include/mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"],
    copts = ["-std=c++14"],
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
        "lib/Conversion/GPUToSPIRV/GPUToSPIRV.cpp",
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":GPUDialect",
        ":IR",
        ":Pass",
        ":SPIRVConversions",
        ":SPIRVDialect",
        ":StandardOps",
    ],
    alwayslink = 1,
)

gentbl(
    name = "LLVMOpsIncGen",
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
        "include/mlir/Dialect/LLVMIR/NVVMOps.cpp.inc",
        "include/mlir/Dialect/LLVMIR/NVVMOps.h.inc",
        "lib/Dialect/LLVMIR/IR/NVVMDialect.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/LLVMIR/NVVMDialect.h",
    ],
    copts = ["-std=c++14"],
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

filegroup(
    name = "SPIRVOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/SPIRV/SPIRVBase.td",
        "include/mlir/Dialect/SPIRV/SPIRVOps.td",
        "include/mlir/Dialect/SPIRV/SPIRVStructureOps.td",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "SPIRVOpsIncGen",
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
    name = "StandardToSPIRVGen",
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
    name = "SPIRVOpUtilsIncGen",
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
        "include/mlir/Dialect/SPIRV/SPIRVEnums.cpp.inc",
        "include/mlir/Dialect/SPIRV/SPIRVEnums.h.inc",
        "include/mlir/Dialect/SPIRV/SPIRVOpUtils.inc",
        "include/mlir/Dialect/SPIRV/SPIRVOps.cpp.inc",
        "include/mlir/Dialect/SPIRV/SPIRVOps.h.inc",
        "lib/Dialect/SPIRV/SPIRVDialect.cpp",
        "lib/Dialect/SPIRV/SPIRVOps.cpp",
        "lib/Dialect/SPIRV/SPIRVTypes.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/SPIRV/SPIRVDialect.h",
        "include/mlir/Dialect/SPIRV/SPIRVOps.h",
        "include/mlir/Dialect/SPIRV/SPIRVTypes.h",
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":IR",
        ":Parser",
        ":SPIRVOpUtilsIncGen",
        ":SPIRVOpsIncGen",
        ":Support",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "SPIRVConversions",
    srcs = [
        "lib/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.cpp",
        "lib/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.cpp",
        "lib/Conversion/StandardToSPIRV/StandardToSPIRV.cpp.inc",
    ],
    hdrs = [
        "include/mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h",
        "include/mlir/Dialect/SPIRV/Passes.h",
    ],
    copts = ["-std=c++14"],
    includes = [
        "include",
        "lib/Conversion/StandardToSPIRV",
    ],
    deps = [
        ":IR",
        ":Pass",
        ":SPIRVDialect",
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
        "include/mlir/Dialect/SPIRV/SPIRVSerialization.inc",
        "lib/Dialect/SPIRV/Serialization/Deserializer.cpp",
        "lib/Dialect/SPIRV/Serialization/SPIRVBinaryUtils.cpp",
        "lib/Dialect/SPIRV/Serialization/Serializer.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/SPIRV/SPIRVBinaryUtils.h",
        "include/mlir/Dialect/SPIRV/Serialization.h",
    ],
    copts = ["-std=c++14"],
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
    name = "SerializeSPIRV",
    srcs = [
        "lib/Dialect/SPIRV/Serialization/ConvertToBinary.cpp",
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":IR",
        ":SPIRVDialect",
        ":SPIRVSerialization",
        ":Support",
        ":Translation",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "DeserializeSPIRV",
    srcs = [
        "lib/Dialect/SPIRV/Serialization/ConvertFromBinary.cpp",
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":IR",
        ":SPIRVDialect",
        ":SPIRVSerialization",
        ":StandardOps",
        ":Support",
        ":Translation",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "SPIRVDialectRegistration",
    srcs = ["lib/Dialect/SPIRV/DialectRegistration.cpp"],
    copts = ["-std=c++14"],
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
        "lib/Transforms/Utils/LoopFusionUtils.cpp",
        "lib/Transforms/Utils/LoopUtils.cpp",
        "lib/Transforms/Utils/RegionUtils.cpp",
        "lib/Transforms/Utils/Utils.cpp",
    ],
    hdrs = [
        "include/mlir/Transforms/FoldUtils.h",
        "include/mlir/Transforms/LoopFusionUtils.h",
        "include/mlir/Transforms/LoopUtils.h",
        "include/mlir/Transforms/RegionUtils.h",
        "include/mlir/Transforms/Utils.h",
    ],
    copts = ["-std=c++14"],
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

cc_library(
    name = "Transforms",
    srcs = [
        "lib/Transforms/AffineDataCopyGeneration.cpp",
        "lib/Transforms/CSE.cpp",
        "lib/Transforms/Canonicalizer.cpp",
        "lib/Transforms/DialectConversion.cpp",
        "lib/Transforms/LoopCoalescing.cpp",
        "lib/Transforms/LoopFusion.cpp",
        "lib/Transforms/LoopInvariantCodeMotion.cpp",
        "lib/Transforms/LoopTiling.cpp",
        "lib/Transforms/LoopUnroll.cpp",
        "lib/Transforms/LoopUnrollAndJam.cpp",
        "lib/Transforms/LowerAffine.cpp",
        "lib/Transforms/LowerVectorTransfers.cpp",
        "lib/Transforms/MaterializeVectors.cpp",
        "lib/Transforms/MemRefDataFlowOpt.cpp",
        "lib/Transforms/PipelineDataTransfer.cpp",
        "lib/Transforms/SimplifyAffineStructures.cpp",
        "lib/Transforms/StripDebugInfo.cpp",
        "lib/Transforms/Vectorize.cpp",
    ],
    hdrs = [
        "include/mlir/Transforms/DialectConversion.h",
        "include/mlir/Transforms/LowerAffine.h",
        "include/mlir/Transforms/Passes.h",
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":Analysis",
        ":EDSC",
        ":IR",
        ":LoopOps",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":VectorOps",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "LoopsToGPU",
    srcs = [
        "lib/Conversion/LoopsToGPU/LoopsToGPU.cpp",
    ],
    hdrs = [
        "include/mlir/Conversion/LoopsToGPU/LoopsToGPU.h",
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":AffineOps",
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
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":LoopOps",
        ":LoopsToGPU",
        ":Pass",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "CFGTransforms",
    srcs = [
        "lib/Conversion/ControlFlowToCFG/ConvertControlFlowToCFG.cpp",
    ],
    hdrs = [
        "include/mlir/Conversion/ControlFlowToCFG/ConvertControlFlowToCFG.h",
    ],
    copts = ["-std=c++14"],
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
        "@llvm//:core",
        "@llvm//:support",
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
    copts = ["-std=c++14"],
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

cc_library(
    name = "Analysis",
    srcs = [
        "lib/Analysis/AffineAnalysis.cpp",
        "lib/Analysis/AffineStructures.cpp",
        "lib/Analysis/Dominance.cpp",
        "lib/Analysis/LoopAnalysis.cpp",
        "lib/Analysis/MemRefBoundCheck.cpp",
        "lib/Analysis/NestedMatcher.cpp",
        "lib/Analysis/OpStats.cpp",
        "lib/Analysis/SliceAnalysis.cpp",
        "lib/Analysis/TestMemRefDependenceCheck.cpp",
        "lib/Analysis/TestParallelismDetection.cpp",
        "lib/Analysis/Utils.cpp",
        "lib/Analysis/VectorAnalysis.cpp",
        "lib/Analysis/Verifier.cpp",
    ],
    hdrs = [
        "include/mlir/Analysis/AffineAnalysis.h",
        "include/mlir/Analysis/AffineStructures.h",
        "include/mlir/Analysis/Dominance.h",
        "include/mlir/Analysis/LoopAnalysis.h",
        "include/mlir/Analysis/NestedMatcher.h",
        "include/mlir/Analysis/Passes.h",
        "include/mlir/Analysis/SliceAnalysis.h",
        "include/mlir/Analysis/Utils.h",
        "include/mlir/Analysis/VectorAnalysis.h",
        "include/mlir/Analysis/Verifier.h",
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":IR",
        ":LoopOps",
        ":Pass",
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
    copts = ["-std=c++14"],
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
    copts = ["-std=c++14"],
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
        "lib/Target/LLVMIR/ConvertToLLVMIR.cpp",
    ],
    hdrs = [
        "include/mlir/Target/LLVMIR.h",
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":LLVMIRModuleTranslation",
        ":Support",
        ":Translation",
        "@llvm//:core",
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
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":GPUDialect",
        ":IR",
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
    name = "ExecutionEngine",
    srcs = [
        "lib/ExecutionEngine/ExecutionEngine.cpp",
        "lib/ExecutionEngine/MemRefUtils.cpp",
    ],
    hdrs = [
        "include/mlir/ExecutionEngine/ExecutionEngine.h",
        "include/mlir/ExecutionEngine/MemRefUtils.h",
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":Support",
        ":TargetLLVMIR",
        ":Translation",
        "@llvm//:core",
        "@llvm//:execution_engine",
        "@llvm//:mc",
        "@llvm//:orc_jit",
        "@llvm//:support",
        "@llvm//:target",  # fixdeps: keep
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
    copts = ["-std=c++14"],
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
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":GPUToNVVMTransforms",
        ":GPUToSPIRVTransforms",
        ":GPUTransforms",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":NVVMDialect",
        ":Parser",
        ":Pass",
        ":QuantizerTransforms",
        ":SPIRVConversions",
        ":SPIRVDialectRegistration",
        ":Support",
        ":Transforms",
        ":VectorToLLVMTransforms",
        ":ViewRegionGraph",
        "@llvm//:support",
    ],
)

cc_library(
    name = "ViewRegionGraph",
    srcs = ["lib/Transforms/ViewRegionGraph.cpp"],
    hdrs = ["include/mlir/Transforms/ViewRegionGraph.h"],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":IR",
        ":Pass",
        ":Support",
        "@llvm//:support",
    ],
)

cc_library(
    name = "TranslateClParser",
    srcs = ["lib/Support/TranslateClParser.cpp"],
    hdrs = ["include/mlir/Support/TranslateClParser.h"],
    copts = ["-std=c++14"],
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

cc_binary(
    name = "mlir-translate",
    copts = ["-std=c++14"],
    deps = [
        ":tools/mlir-translate/mlir-translate",
    ],
)

cc_library(
    name = "tools/mlir-translate/mlir-translate",
    srcs = ["tools/mlir-translate/mlir-translate.cpp"],
    copts = ["-std=c++14"],
    deps = [
        ":DeserializeSPIRV",
        ":IR",
        ":LoopDialectRegistration",
        ":Parser",
        ":SPIRVDialectRegistration",
        ":SerializeSPIRV",
        ":StandardDialectRegistration",
        ":Support",
        ":TargetLLVMIR",
        ":TargetNVVMIR",
        ":TranslateClParser",
        ":Translation",
        ":VectorDialectRegistration",
        "@llvm//:support",
    ],
)

# TODO(jpienaar): This library should be removed.
cc_library(
    name = "MlirOptMain",
    srcs = [
        "tools/mlir-opt/mlir-opt.cpp",
    ],
    copts = ["-std=c++14"],
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
    copts = ["-std=c++14"],
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
        ":StandardDialectRegistration",
        ":Transforms",
        ":VectorDialectRegistration",
        "//test:TestDialect",
        "//test:TestTransforms",
        "@llvm//:support",
    ],
)

cc_library(
    name = "MlirJitRunner",
    srcs = ["lib/Support/JitRunner.cpp"],
    hdrs = ["include/mlir/Support/JitRunner.h"],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":ExecutionEngine",
        ":ExecutionEngineUtils",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":LoopDialectRegistration",
        ":Parser",
        ":Pass",
        ":StandardDialectRegistration",
        ":Support",
        ":Transforms",
        "@llvm//:core",
        "@llvm//:orc_jit",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "mlir-cpu-runner",
    srcs = ["tools/mlir-cpu-runner/mlir-cpu-runner.cpp"],
    copts = ["-std=c++14"],
    linkopts = ["-ldl"],
    deps = [
        ":MlirJitRunner",
        "@llvm//:support",
    ],
)

cc_binary(
    name = "tools/libcuda-runtime-wrappers.so",
    srcs = ["tools/mlir-cuda-runner/cuda-runtime-wrappers.cpp"],
    copts = ["-std=c++14"],
    includes = ["include"],
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
    copts = ["-std=c++14"],
    data = [
        ":tools/libcuda-runtime-wrappers.so",
    ],
    includes = ["include"],
    deps = [
        ":GPUDialect",
        ":GPUDialectRegistration",
        ":GPUToNVVMTransforms",
        ":GPUTransforms",
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":MlirJitRunner",
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
        "include/mlir/TableGen/OpTrait.h",
        "include/mlir/TableGen/Operator.h",
        "include/mlir/TableGen/Pattern.h",
        "include/mlir/TableGen/Predicate.h",
        "include/mlir/TableGen/Region.h",
        "include/mlir/TableGen/Type.h",
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":Support",
        "@llvm//:support",
        "@llvm//:tablegen",
    ],
)

cc_binary(
    name = "mlir-tblgen",
    srcs = [
        "tools/mlir-tblgen/EnumsGen.cpp",
        "tools/mlir-tblgen/LLVMIRConversionGen.cpp",
        "tools/mlir-tblgen/OpDefinitionsGen.cpp",
        "tools/mlir-tblgen/OpDocGen.cpp",
        "tools/mlir-tblgen/ReferenceImplGen.cpp",
        "tools/mlir-tblgen/RewriterGen.cpp",
        "tools/mlir-tblgen/SPIRVUtilsGen.cpp",
        "tools/mlir-tblgen/mlir-tblgen.cpp",
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    linkopts = [
        "-lm",
        "-lpthread",
    ],
    deps = [
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
        "include/mlir/Dialect/QuantOps/QuantOps.cpp.inc",
        "include/mlir/Dialect/QuantOps/QuantOps.h.inc",
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
    copts = ["-std=c++14"],
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
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":IR",
        ":QuantOps",
        "@llvm//:support",
    ],
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
        "include/mlir/Dialect/FxpMathOps/FxpMathOps.cpp.inc",
        "include/mlir/Dialect/FxpMathOps/FxpMathOps.h.inc",
        "lib/Dialect/FxpMathOps/IR/FxpMathOps.cpp",
        "lib/Dialect/FxpMathOps/Transforms/LowerUniformRealMath.cpp",
        "lib/Dialect/FxpMathOps/Transforms/UniformKernelUtils.h",
    ],
    hdrs = [
        "include/mlir/Dialect/FxpMathOps/FxpMathOps.h",
        "include/mlir/Dialect/FxpMathOps/Passes.h",
    ],
    copts = ["-std=c++14"],
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
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":FxpMathOps",
        ":IR",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

filegroup(
    name = "LinalgOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Linalg/IR/LinalgBase.td",
        "include/mlir/Dialect/Linalg/IR/LinalgOps.td",
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "LinalgOpsIncGen",
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
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/Linalg/IR/LinalgLibraryOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/Linalg/IR/LinalgLibraryOps.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/IR/LinalgLibraryOps.td",
    td_srcs = [
        ":LinalgLibraryOpsTdFiles",
    ],
)

cc_library(
    name = "Linalg",
    srcs = [
        "lib/Dialect/Linalg/Analysis/DependenceAnalysis.cpp",
        "lib/Dialect/Linalg/IR/LinalgOps.cpp",
        "lib/Dialect/Linalg/IR/LinalgTypes.cpp",
        "lib/Dialect/Linalg/Transforms/Fusion.cpp",
        "lib/Dialect/Linalg/Transforms/LowerToLLVMDialect.cpp",
        "lib/Dialect/Linalg/Transforms/LowerToLoops.cpp",
        "lib/Dialect/Linalg/Transforms/Tiling.cpp",
        "lib/Dialect/Linalg/Utils/Utils.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h",
        "include/mlir/Dialect/Linalg/IR/LinalgOps.h",
        "include/mlir/Dialect/Linalg/IR/LinalgTraits.h",
        "include/mlir/Dialect/Linalg/IR/LinalgTypes.h",
        "include/mlir/Dialect/Linalg/Passes.h",
        "include/mlir/Dialect/Linalg/Utils/Intrinsics.h",
        "include/mlir/Dialect/Linalg/Utils/Utils.h",
    ],
    copts = ["-std=c++14"],
    deps = [
        ":AffineOps",
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
        ":VectorToLLVMTransforms",
        "@llvm//:core",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "LinalgDialectRegistration",
    srcs = ["lib/Dialect/Linalg/LinalgRegistration.cpp"],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":IR",
        ":Linalg",
        "@llvm//:support",
    ],
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
    copts = ["-std=c++14"],
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
    copts = ["-std=c++14"],
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
        ":OpBaseTdFiles",
    ],
)

gentbl(
    name = "VectorOpsIncGen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/mlir/Dialect/VectorOps/VectorOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/mlir/Dialect/VectorOps/VectorOps.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/VectorOps/VectorOps.td",
    td_srcs = [
        ":VectorOpsTdFiles",
    ],
)

cc_library(
    name = "VectorToLLVMTransforms",
    srcs = ["lib/Conversion/VectorToLLVM/VectorToLLVM.cpp"],
    hdrs = [
        "include/mlir/Conversion/VectorToLLVM/VectorToLLVM.h",
    ],
    copts = ["-std=c++14"],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":LLVMTransforms",
        ":Pass",
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
