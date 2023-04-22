# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = [":test_friends"],
    licenses = ["notice"],
)

# Please only depend on this from MLIR tests.
package_group(
    name = "test_friends",
    packages = ["//..."],
)

cc_library(
    name = "IRProducingAPITest",
    hdrs = ["APITest.h"],
    includes = ["."],
)

cc_library(
    name = "TestAnalysis",
    srcs = glob(["lib/Analysis/*.cpp"]),
    includes = ["lib/Dialect/Test"],
    deps = [
        ":TestDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "TestOpTdFiles",
    srcs = [
        "lib/Dialect/Test/TestInterfaces.td",
        "lib/Dialect/Test/TestOps.td",
        "@llvm-project//mlir:include/mlir/Dialect/DLTI/DLTIBase.td",
        "@llvm-project//mlir:include/mlir/IR/OpAsmInterface.td",
        "@llvm-project//mlir:include/mlir/IR/RegionKindInterface.td",
        "@llvm-project//mlir:include/mlir/IR/SymbolInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/CallInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/ControlFlowInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/CopyOpInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/DataLayoutInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
    ],
)

gentbl_cc_library(
    name = "TestOpsIncGen",
    strip_include_prefix = "lib/Dialect/Test",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "lib/Dialect/Test/TestOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "lib/Dialect/Test/TestOps.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=test",
            ],
            "lib/Dialect/Test/TestOpsDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=test",
            ],
            "lib/Dialect/Test/TestOpsDialect.cpp.inc",
        ),
        (
            ["-gen-enum-decls"],
            "lib/Dialect/Test/TestOpEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "lib/Dialect/Test/TestOpEnums.cpp.inc",
        ),
        (
            ["-gen-struct-attr-decls"],
            "lib/Dialect/Test/TestOpStructs.h.inc",
        ),
        (
            ["-gen-struct-attr-defs"],
            "lib/Dialect/Test/TestOpStructs.cpp.inc",
        ),
        (
            ["-gen-rewriters"],
            "lib/Dialect/Test/TestPatterns.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/Test/TestOps.td",
    test = True,
    deps = [
        ":TestOpTdFiles",
    ],
)

gentbl_cc_library(
    name = "TestInterfacesIncGen",
    strip_include_prefix = "lib/Dialect/Test",
    tbl_outs = [
        (
            ["-gen-attr-interface-decls"],
            "lib/Dialect/Test/TestAttrInterfaces.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "lib/Dialect/Test/TestAttrInterfaces.cpp.inc",
        ),
        (
            ["-gen-type-interface-decls"],
            "lib/Dialect/Test/TestTypeInterfaces.h.inc",
        ),
        (
            ["-gen-type-interface-defs"],
            "lib/Dialect/Test/TestTypeInterfaces.cpp.inc",
        ),
        (
            ["-gen-op-interface-decls"],
            "lib/Dialect/Test/TestOpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "lib/Dialect/Test/TestOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/Test/TestInterfaces.td",
    test = True,
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "TestAttrDefsIncGen",
    strip_include_prefix = "lib/Dialect/Test",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "lib/Dialect/Test/TestAttrDefs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "lib/Dialect/Test/TestAttrDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/Test/TestAttrDefs.td",
    td_srcs = [
        ":TestOpTdFiles",
    ],
    test = True,
)

gentbl_cc_library(
    name = "TestTypeDefsIncGen",
    strip_include_prefix = "lib/Dialect/Test",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "--typedefs-dialect=test",
            ],
            "lib/Dialect/Test/TestTypeDefs.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "--typedefs-dialect=test",
            ],
            "lib/Dialect/Test/TestTypeDefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/Test/TestTypeDefs.td",
    test = True,
    deps = [
        ":TestOpTdFiles",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
    ],
)

cc_library(
    name = "TestDialect",
    srcs = [
        "lib/Dialect/Test/TestAttributes.cpp",
        "lib/Dialect/Test/TestDialect.cpp",
        "lib/Dialect/Test/TestInterfaces.cpp",
        "lib/Dialect/Test/TestPatterns.cpp",
        "lib/Dialect/Test/TestTraits.cpp",
        "lib/Dialect/Test/TestTypes.cpp",
    ],
    hdrs = [
        "lib/Dialect/Test/TestAttributes.h",
        "lib/Dialect/Test/TestDialect.h",
        "lib/Dialect/Test/TestInterfaces.h",
        "lib/Dialect/Test/TestTypes.h",
    ],
    includes = [
        "lib/Dialect/Test",
    ],
    deps = [
        ":TestAttrDefsIncGen",
        ":TestInterfacesIncGen",
        ":TestOpsIncGen",
        ":TestTypeDefsIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:CopyOpInterface",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Reducer",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "TestIR",
    srcs = glob(["lib/IR/*.cpp"]),
    deps = [
        ":TestDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "TestPass",
    srcs = glob(["lib/Pass/*.cpp"]),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "TestRewrite",
    srcs = [
        "lib/Rewrite/TestPDLByteCode.cpp",
    ],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "TestReducer",
    srcs = [
        "lib/Reducer/MLIRTestReducer.cpp",
    ],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "TestTransforms",
    srcs = glob(["lib/Transforms/*.cpp"]),
    defines = ["MLIR_CUDA_CONVERSIONS_ENABLED"],
    includes = ["lib/Dialect/Test"],
    deps = [
        ":TestDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "TestStandardToLLVM",
    srcs = glob(["lib/Conversion/StandardToLLVM/*.cpp"]),
    defines = ["MLIR_CUDA_CONVERSIONS_ENABLED"],
    includes = ["lib/Dialect/Test"],
    deps = [
        ":TestDialect",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardToLLVM",
    ],
)

cc_library(
    name = "TestAffine",
    srcs = glob([
        "lib/Dialect/Affine/*.cpp",
    ]),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
    ],
)

cc_library(
    name = "TestDLTI",
    srcs = glob(["lib/Dialect/DLTI/*.cpp"]),
    defines = ["MLIR_CUDA_CONVERSIONS_ENABLED"],
    includes = ["lib/Dialect/Test"],
    deps = [
        ":TestDialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "TestGPU",
    srcs = glob(["lib/Dialect/GPU/*.cpp"]),
    defines = ["MLIR_CUDA_CONVERSIONS_ENABLED"],
    includes = ["lib/Dialect/Test"],
    deps = [
        "@llvm-project//llvm:NVPTXCodeGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLToLLVMIRTranslation",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "TestLinalg",
    srcs = glob(["lib/Dialect/Linalg/*.cpp"]),
    defines = ["MLIR_CUDA_CONVERSIONS_ENABLED"],
    includes = ["lib/Dialect/Test"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:VectorToSCF",
    ],
)

cc_library(
    name = "TestMath",
    srcs = glob(["lib/Dialect/Math/*.cpp"]),
    defines = ["MLIR_CUDA_CONVERSIONS_ENABLED"],
    includes = ["lib/Dialect/Test"],
    deps = [
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorOps",
    ],
)

cc_library(
    name = "TestSCF",
    srcs = glob(["lib/Dialect/SCF/*.cpp"]),
    defines = ["MLIR_CUDA_CONVERSIONS_ENABLED"],
    includes = ["lib/Dialect/Test"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "TestShapeDialect",
    srcs = [
        "lib/Dialect/Shape/TestShapeFunctions.cpp",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
    ],
)

cc_library(
    name = "TestSPIRV",
    srcs = glob([
        "lib/Dialect/SPIRV/*.cpp",
    ]),
    deps = [
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVConversion",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:SPIRVModuleCombiner",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "TestStandardOps",
    srcs = glob(["lib/Dialect/StandardOps/*.cpp"]),
    defines = ["MLIR_CUDA_CONVERSIONS_ENABLED"],
    includes = ["lib/Dialect/Test"],
    deps = [
        ":TestDialect",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "TestVector",
    srcs = glob(["lib/Dialect/Vector/*.cpp"]),
    defines = ["MLIR_CUDA_CONVERSIONS_ENABLED"],
    includes = ["lib/Dialect/Test"],
    deps = [
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:VectorToSCF",
    ],
)

cc_library(
    name = "TestTypeDialect",
    srcs = glob([
        "lib/Dialect/LLVMIR/*.cpp",
    ]),
    deps = [
        ":TestDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
    ],
)

cc_library(
    name = "TestTosaDialect",
    srcs = glob([
        "lib/Dialect/Tosa/*.cpp",
    ]),
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:Transforms",
    ],
)
