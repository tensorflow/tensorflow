# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Description:
#   The MLIR "Multi-Level Intermediate Representation" Compiler Infrastructure

load("@org_tensorflow//third_party/mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@org_tensorflow//third_party/mlir:linalggen.bzl", "genlinalg")
load("@org_tensorflow//third_party/mlir:build_defs.bzl", "cc_headers_only", "if_cuda_available")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

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

filegroup(
    name = "c_headers",
    srcs = glob(["include/mlir-c/**/*"]),  # <== i.e. match the entire tree
    visibility = [
        ":friends",
    ],
)

[
    gentbl_cc_library(
        name = name + "IncGen",
        strip_include_prefix = "include",
        tbl_outs = [
            (
                ["-gen-op-interface-decls"],
                "include/mlir/IR/" + name + ".h.inc",
            ),
            (
                ["-gen-op-interface-defs"],
                "include/mlir/IR/" + name + ".cpp.inc",
            ),
        ],
        tblgen = ":mlir-tblgen",
        td_file = "include/mlir/IR/" + name + ".td",
        deps = [":OpBaseTdFiles"],
    )
    for name in [
        "OpAsmInterface",
        "RegionKindInterface",
        "SymbolInterfaces",
    ]
]

gentbl_cc_library(
    name = "TensorEncodingIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-attr-interface-decls"],
            "include/mlir/IR/TensorEncInterfaces.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "include/mlir/IR/TensorEncInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/IR/TensorEncoding.td",
    deps = [":TensorOpsTdFiles"],
)

td_library(
    name = "BuiltinDialectTdFiles",
    srcs = [
        "include/mlir/IR/BuiltinDialect.td",
        "include/mlir/IR/BuiltinLocationAttributes.td",
        "include/mlir/IR/BuiltinOps.td",
        "include/mlir/IR/BuiltinTypes.td",
    ],
    includes = ["include"],
    deps = [
        ":CallInterfacesTdFiles",
        ":CastInterfacesTdFiles",
        ":DataLayoutInterfacesTdFiles",
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "BuiltinDialectIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "include/mlir/IR/BuiltinDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/IR/BuiltinDialect.td",
    deps = [":BuiltinDialectTdFiles"],
)

gentbl_cc_library(
    name = "BuiltinAttributesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["--gen-attrdef-decls"],
            "include/mlir/IR/BuiltinAttributes.h.inc",
        ),
        (
            ["--gen-attrdef-defs"],
            "include/mlir/IR/BuiltinAttributes.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/IR/BuiltinAttributes.td",
    deps = [":BuiltinDialectTdFiles"],
)

gentbl_cc_library(
    name = "BuiltinLocationAttributesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["--gen-attrdef-decls"],
            "include/mlir/IR/BuiltinLocationAttributes.h.inc",
        ),
        (
            ["--gen-attrdef-defs"],
            "include/mlir/IR/BuiltinLocationAttributes.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/IR/BuiltinLocationAttributes.td",
    deps = [":BuiltinDialectTdFiles"],
)

gentbl_cc_library(
    name = "BuiltinOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/IR/BuiltinOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/IR/BuiltinOps.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/IR/BuiltinOps.td",
    deps = [":BuiltinDialectTdFiles"],
)

gentbl_cc_library(
    name = "BuiltinTypesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["--gen-typedef-decls"],
            "include/mlir/IR/BuiltinTypes.h.inc",
        ),
        (
            ["--gen-typedef-defs"],
            "include/mlir/IR/BuiltinTypes.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/IR/BuiltinTypes.td",
    deps = [":BuiltinDialectTdFiles"],
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
        "include/mlir/Interfaces/CallInterfaces.h",
        "include/mlir/Interfaces/CastInterfaces.h",
        "include/mlir/Interfaces/SideEffectInterfaces.h",
        "include/mlir/Interfaces/DataLayoutInterfaces.h",
        "include/mlir/Interfaces/DecodeAttributesInterfaces.h",
        "include/mlir/Interfaces/FoldInterfaces.h",
    ],
    includes = ["include"],
    deps = [
        ":BuiltinAttributesIncGen",
        ":BuiltinDialectIncGen",
        ":BuiltinLocationAttributesIncGen",
        ":BuiltinOpsIncGen",
        ":BuiltinTypesIncGen",
        ":CallOpInterfacesIncGen",
        ":CastOpInterfacesIncGen",
        ":DataLayoutInterfacesIncGen",
        ":InferTypeOpInterfaceIncGen",
        ":OpAsmInterfaceIncGen",
        ":RegionKindInterfaceIncGen",
        ":SideEffectInterfacesIncGen",
        ":Support",
        ":SymbolInterfacesIncGen",
        ":TensorEncodingIncGen",
        "@llvm-project//llvm:Support",
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
    ]),
    includes = ["include"],
    linkopts = [
        "-lm",
        "-lpthread",
    ],
    deps = [
        ":IR",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

# TODO(ntv): Update these to enable simplifying the cmake and build files.
cc_library(
    name = "EDSC",
    srcs = ["lib/EDSC/Builders.cpp"],
    hdrs = ["include/mlir/EDSC/Builders.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "CAPIIR",
    srcs = [
        "lib/CAPI/Dialect/Standard.cpp",
        "lib/CAPI/IR/AffineExpr.cpp",
        "lib/CAPI/IR/AffineMap.cpp",
        "lib/CAPI/IR/BuiltinAttributes.cpp",
        "lib/CAPI/IR/BuiltinTypes.cpp",
        "lib/CAPI/IR/Diagnostics.cpp",
        "lib/CAPI/IR/DialectHandle.cpp",
        "lib/CAPI/IR/IR.cpp",
        "lib/CAPI/IR/IntegerSet.cpp",
        "lib/CAPI/IR/Pass.cpp",
        "lib/CAPI/IR/Support.cpp",
    ],
    hdrs = [
        "include/mlir-c/AffineExpr.h",
        "include/mlir-c/AffineMap.h",
        "include/mlir-c/BuiltinAttributes.h",
        "include/mlir-c/BuiltinTypes.h",
        "include/mlir-c/Diagnostics.h",
        "include/mlir-c/Dialect/Standard.h",
        "include/mlir-c/ExecutionEngine.h",
        "include/mlir-c/IR.h",
        "include/mlir-c/IntegerSet.h",
        "include/mlir-c/Pass.h",
        "include/mlir-c/Registration.h",
        "include/mlir-c/Support.h",
        "include/mlir/CAPI/AffineExpr.h",
        "include/mlir/CAPI/AffineMap.h",
        "include/mlir/CAPI/Diagnostics.h",
        "include/mlir/CAPI/IR.h",
        "include/mlir/CAPI/IntegerSet.h",
        "include/mlir/CAPI/Pass.h",
        "include/mlir/CAPI/Registration.h",
        "include/mlir/CAPI/Support.h",
        "include/mlir/CAPI/Utils.h",
        "include/mlir/CAPI/Wrap.h",
    ],
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":InferTypeOpInterface",
        ":Parser",
        ":Pass",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "CAPIAsync",
    srcs = [
        "lib/CAPI/Dialect/Async.cpp",
        "lib/CAPI/Dialect/AsyncPasses.cpp",
    ],
    hdrs = [
        "include/mlir-c/Dialect/Async.h",
    ],
    includes = ["include"],
    deps = [
        ":Async",
        ":AsyncPassIncGen",
        ":AsyncTransforms",
        ":CAPIIR",
        ":Pass",
    ],
)

cc_library(
    name = "CAPILinalg",
    srcs = [
        "lib/CAPI/Dialect/Linalg.cpp",
        "lib/CAPI/Dialect/LinalgPasses.cpp",
    ],
    hdrs = [
        "include/mlir-c/Dialect/Linalg.h",
    ],
    includes = ["include"],
    deps = [
        ":CAPIIR",
        ":LinalgOps",
        ":LinalgPassIncGen",
        ":LinalgTransforms",
        ":Pass",
    ],
)

cc_library(
    name = "CAPIGPU",
    srcs = [
        "lib/CAPI/Dialect/GPU.cpp",
        "lib/CAPI/Dialect/GPUPasses.cpp",
    ],
    hdrs = [
        "include/mlir-c/Dialect/GPU.h",
    ],
    includes = ["include"],
    deps = [
        ":CAPIIR",
        ":GPUDialect",
        ":GPUPassIncGen",
        ":GPUTransforms",
        ":Pass",
    ],
)

cc_library(
    name = "CAPISparseTensor",
    srcs = [
        "lib/CAPI/Dialect/SparseTensor.cpp",
        "lib/CAPI/Dialect/SparseTensorPasses.cpp",
    ],
    hdrs = [
        "include/mlir-c/Dialect/SparseTensor.h",
    ],
    includes = ["include"],
    deps = [
        ":CAPIIR",
        ":Pass",
        ":SparseTensor",
        ":SparseTensorPassIncGen",
        ":SparseTensorTransforms",
        ":Support",
    ],
)

cc_library(
    name = "CAPIConversion",
    srcs = ["lib/CAPI/Conversion/Passes.cpp"],
    hdrs = ["include/mlir-c/Conversion.h"],
    includes = ["include"],
    deps = [
        ":CAPIIR",
        ":ConversionPassIncGen",
        ":ConversionPasses",
        ":Pass",
    ],
)

cc_library(
    name = "CAPIDebug",
    srcs = ["lib/CAPI/Debug/Debug.cpp"],
    hdrs = ["include/mlir-c/Debug.h"],
    includes = ["include"],
    deps = [
        ":CAPIIR",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "CAPIExecutionEngine",
    srcs = ["lib/CAPI/ExecutionEngine/ExecutionEngine.cpp"],
    hdrs = [
        "include/mlir-c/ExecutionEngine.h",
        "include/mlir/CAPI/ExecutionEngine.h",
    ],
    includes = ["include"],
    deps = [
        ":CAPIIR",
        ":ExecutionEngine",
        ":ExecutionEngineUtils",
        ":LLVMToLLVMIRTranslation",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "CAPITransforms",
    srcs = ["lib/CAPI/Transforms/Passes.cpp"],
    hdrs = ["include/mlir-c/Transforms.h"],
    includes = ["include"],
    deps = [
        ":CAPIIR",
        ":Pass",
        ":Transforms",
        ":TransformsPassIncGen",
    ],
)

cc_library(
    name = "CAPIRegistration",
    srcs = ["lib/CAPI/Registration/Registration.cpp"],
    hdrs = ["include/mlir-c/Registration.h"],
    includes = ["include"],
    deps = [
        ":AllPassesAndDialects",
        ":CAPIIR",
        ":LLVMToLLVMIRTranslation",
    ],
)

##---------------------------------------------------------------------------##
# Sources of Python bindings.
#----------------------------------------------------------------------------##

exports_files(
    glob(["lib/Bindings/Python/**/*.cpp"]),
)

cc_library(
    name = "MLIRBindingsPythonHeadersAndDeps",
    includes = [
        "include",
        "lib/Bindings/Python",
    ],
    textual_hdrs = glob([
        "lib/Bindings/Python/*.h",
        "include/mlir-c/Bindings/Python/*.h",
        "include/mlir/Bindings/Python/*.h",
    ]),
    deps = [
        ":CAPIIR",
        "//third_party/pybind11",
        "//third_party/python_runtime:headers",
    ],
)

cc_library(
    name = "MLIRBindingsPythonCore",
    srcs = glob(
        include = ["lib/Bindings/Python/*.cpp"],
        exclude = ["lib/Bindings/Python/MainModule.cpp"],
    ),
    # These flags are needed for pybind11 to work.
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = [
        # Cannot use header_modules (parse_headers feature fails).
        "-use_header_modules",
    ],
    deps = [
        ":CAPIAsync",
        ":CAPIDebug",
        ":CAPIExecutionEngine",
        ":CAPIGPU",
        ":CAPIIR",
        ":CAPILinalg",
        ":CAPIRegistration",
        ":CAPISparseTensor",
        ":MLIRBindingsPythonHeadersAndDeps",
        "//third_party/pybind11",
        "//third_party/python_runtime:headers",
        "@llvm-project//llvm:Support",
    ],
)

PYBIND11_COPTS = [
    "-fexceptions",
    "-frtti",
]

PYBIND11_FEATURES = [
    # Cannot use header_modules (parse_headers feature fails).
    "-use_header_modules",
]

# Dynamic library with the MLIR Python extension.
cc_binary(
    name = "_mlir.so",
    srcs = ["lib/Bindings/Python/MainModule.cpp"],
    # These flags are needed for pybind11 to work.
    copts = PYBIND11_COPTS,
    features = PYBIND11_FEATURES,
    linkshared = 1,
    linkstatic = 0,
    deps = [
        ":MLIRBindingsPythonCore",
        ":MLIRBindingsPythonHeadersAndDeps",
    ],
)

# Dynamlic library with the MLIR Conversions Python extension.
cc_binary(
    name = "_mlirConversions.so",
    srcs = ["lib/Bindings/Python/Conversions/Conversions.cpp"],
    copts = PYBIND11_COPTS,
    features = PYBIND11_FEATURES,
    linkshared = 1,
    linkstatic = 0,
    deps = [
        ":CAPIConversion",
        ":MLIRBindingsPythonHeadersAndDeps",
        "//third_party/pybind11",
        "//third_party/python_runtime:headers",
    ],
)

# Dynamlic library with the MLIR Transforms Python extension.
cc_binary(
    name = "_mlirTransforms.so",
    srcs = ["lib/Bindings/Python/Transforms/Transforms.cpp"],
    copts = PYBIND11_COPTS,
    features = PYBIND11_FEATURES,
    linkshared = 1,
    linkstatic = 0,
    deps = [
        ":CAPITransforms",
        ":MLIRBindingsPythonHeadersAndDeps",
        "//third_party/pybind11",
        "//third_party/python_runtime:headers",
    ],
)

# Dynamic library with the MLIR Linalg dialect+passes Python extension.
cc_binary(
    name = "_mlirLinalgPasses.so",
    srcs = ["lib/Bindings/Python/LinalgPasses.cpp"],
    copts = PYBIND11_COPTS,
    features = PYBIND11_FEATURES,
    linkshared = 1,
    linkstatic = 0,
    deps = [
        ":CAPILinalg",
        ":MLIRBindingsPythonHeadersAndDeps",
        "//third_party/pybind11",
        "//third_party/python_runtime:headers",
    ],
)

##---------------------------------------------------------------------------##

td_library(
    name = "OpBaseTdFiles",
    srcs = [
        "include/mlir/IR/OpAsmInterface.td",
        "include/mlir/IR/OpBase.td",
        "include/mlir/IR/RegionKindInterface.td",
        "include/mlir/IR/SymbolInterfaces.td",
        "include/mlir/IR/TensorEncoding.td",
    ],
    includes = ["include"],
)

td_library(
    name = "CallInterfacesTdFiles",
    srcs = ["include/mlir/Interfaces/CallInterfaces.td"],
    includes = ["include"],
    deps = [":OpBaseTdFiles"],
)

td_library(
    name = "CastInterfacesTdFiles",
    srcs = ["include/mlir/Interfaces/CastInterfaces.td"],
    includes = ["include"],
    deps = [":OpBaseTdFiles"],
)

td_library(
    name = "ControlFlowInterfacesTdFiles",
    srcs = ["include/mlir/Interfaces/ControlFlowInterfaces.td"],
    includes = ["include"],
    deps = [":OpBaseTdFiles"],
)

td_library(
    name = "CopyOpInterfaceTdFiles",
    srcs = ["include/mlir/Interfaces/CopyOpInterface.td"],
    includes = ["include"],
    deps = [":OpBaseTdFiles"],
)

td_library(
    name = "DerivedAttributeOpInterfaceTdFiles",
    srcs = ["include/mlir/Interfaces/DerivedAttributeOpInterface.td"],
    includes = ["include"],
    deps = [":OpBaseTdFiles"],
)

td_library(
    name = "InferTypeOpInterfaceTdFiles",
    srcs = ["include/mlir/Interfaces/InferTypeOpInterface.td"],
    includes = ["include"],
    deps = [":OpBaseTdFiles"],
)

td_library(
    name = "LoopLikeInterfaceTdFiles",
    srcs = ["include/mlir/Interfaces/LoopLikeInterface.td"],
    includes = ["include"],
    deps = [":OpBaseTdFiles"],
)

td_library(
    name = "SideEffectInterfacesTdFiles",
    srcs = [
        "include/mlir/Interfaces/SideEffectInterfaceBase.td",
        "include/mlir/Interfaces/SideEffectInterfaces.td",
    ],
    includes = ["include"],
    deps = [":OpBaseTdFiles"],
)

alias(
    name = "SideEffectTdFiles",
    actual = ":SideEffectInterfacesTdFiles",
)

td_library(
    name = "VectorInterfacesTdFiles",
    srcs = ["include/mlir/Interfaces/VectorInterfaces.td"],
    includes = ["include"],
    deps = [":OpBaseTdFiles"],
)

td_library(
    name = "ViewLikeInterfaceTdFiles",
    srcs = ["include/mlir/Interfaces/ViewLikeInterface.td"],
    includes = ["include"],
    deps = [":OpBaseTdFiles"],
)

td_library(
    name = "ReducerTdFiles",
    srcs = ["include/mlir/Reducer/Passes.td"],
    includes = ["include"],
    deps = [":OpBaseTdFiles"],
)

##---------------------------------------------------------------------------##
# Affine dialect.
##---------------------------------------------------------------------------##

td_library(
    name = "PassBaseTdFiles",
    srcs = ["include/mlir/Pass/PassBase.td"],
    includes = ["include"],
)

td_library(
    name = "AffineOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td",
        "include/mlir/Dialect/Affine/IR/AffineOps.td",
    ],
    includes = ["include"],
    deps = [
        ":LoopLikeInterfaceTdFiles",
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
        ":StdOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "AffineOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/Affine/IR/AffineOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/Affine/IR/AffineOps.cpp.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "include/mlir/Dialect/Affine/IR/AffineOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Affine/IR/AffineOps.td",
    deps = [":AffineOpsTdFiles"],
)

gentbl_cc_library(
    name = "AffineMemoryOpInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td",
    deps = [":AffineOpsTdFiles"],
)

##---------------------------------------------------------------------------##
# Async dialect.
##---------------------------------------------------------------------------##

td_library(
    name = "AsyncOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Async/IR/AsyncDialect.td",
        "include/mlir/Dialect/Async/IR/AsyncOps.td",
        "include/mlir/Dialect/Async/IR/AsyncTypes.td",
    ],
    includes = ["include"],
    deps = [
        ":ControlFlowInterfacesTdFiles",
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "AsyncOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/Async/IR/AsyncOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/Async/IR/AsyncOps.cpp.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "include/mlir/Dialect/Async/IR/AsyncOpsDialect.h.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "include/mlir/Dialect/Async/IR/AsyncOpsTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "include/mlir/Dialect/Async/IR/AsyncOpsTypes.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Async/IR/AsyncOps.td",
    deps = [":AsyncOpsTdFiles"],
)

gentbl_cc_library(
    name = "AsyncPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Async",
            ],
            "include/mlir/Dialect/Async/Passes.h.inc",
        ),
        (
            [
                "-gen-pass-capi-header",
                "--prefix=Async",
            ],
            "include/mlir/Dialect/Async/Passes.capi.h.inc",
        ),
        (
            [
                "-gen-pass-capi-impl",
                "--prefix=Async",
            ],
            "include/mlir/Dialect/Async/Passes.capi.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Async/Passes.td",
    deps = [":PassBaseTdFiles"],
)

##---------------------------------------------------------------------------##
# ArmNeon dialect.
##---------------------------------------------------------------------------##

td_library(
    name = "ArmNeonTdFiles",
    srcs = ["include/mlir/Dialect/ArmNeon/ArmNeon.td"],
    includes = ["include"],
    deps = [
        ":LLVMOpsTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "ArmNeonIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=arm_neon",
            ],
            "include/mlir/Dialect/ArmNeon/ArmNeonDialect.h.inc",
        ),
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/ArmNeon/ArmNeon.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/ArmNeon/ArmNeon.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "g3doc/Dialects/ArmNeon/ArmNeon.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/ArmNeon/ArmNeon.td",
    deps = [":ArmNeonTdFiles"],
)

cc_library(
    name = "ArmNeon",
    srcs = ["lib/Dialect/ArmNeon/IR/ArmNeonDialect.cpp"],
    hdrs = ["include/mlir/Dialect/ArmNeon/ArmNeonDialect.h"],
    includes = ["include"],
    deps = [
        ":ArmNeonIncGen",
        ":IR",
        ":SideEffectInterfaces",
        ":VectorOps",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "ArmNeonConversionIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-llvmir-conversions"],
            "include/mlir/Dialect/ArmNeon/ArmNeonConversions.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/ArmNeon/ArmNeon.td",
    deps = [":ArmNeonTdFiles"],
)

##---------------------------------------------------------------------------##
# ArmSVE dialect.
##---------------------------------------------------------------------------##

td_library(
    name = "ArmSVETdFiles",
    srcs = [
        "include/mlir/Dialect/ArmSVE/ArmSVE.td",
        "include/mlir/Dialect/ArmSVE/ArmSVEOpBase.td",
    ],
    includes = ["include"],
    deps = [
        ":LLVMOpsTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "ArmSVEIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/ArmSVE/ArmSVE.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/ArmSVE/ArmSVE.cpp.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "include/mlir/Dialect/ArmSVE/ArmSVETypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "include/mlir/Dialect/ArmSVE/ArmSVETypes.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=arm_sve",
            ],
            "include/mlir/Dialect/ArmSVE/ArmSVEDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/ArmSVE/ArmSVE.td",
    deps = [":ArmSVETdFiles"],
)

cc_library(
    name = "ArmSVE",
    srcs = ["lib/Dialect/ArmSVE/IR/ArmSVEDialect.cpp"],
    hdrs = ["include/mlir/Dialect/ArmSVE/ArmSVEDialect.h"],
    includes = ["include"],
    deps = [
        ":ArmSVEIncGen",
        ":IR",
        ":LLVMDialect",
        ":SideEffectInterfaces",
        ":VectorOps",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "ArmSVETransforms",
    srcs = glob(["lib/Dialect/ArmSVE/Transforms/*.cpp"]),
    hdrs = ["include/mlir/Dialect/ArmSVE/Transforms.h"],
    includes = ["include"],
    deps = [
        ":ArmSVE",
        ":IR",
        ":LLVMDialect",
        ":Pass",
        ":StandardOps",
        ":StandardToLLVM",
        ":TransformUtils",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "ArmSVEConversionIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-llvmir-conversions"],
            "include/mlir/Dialect/ArmSVE/ArmSVEConversions.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/ArmSVE/ArmSVE.td",
    deps = [":ArmSVETdFiles"],
)

##---------------------------------------------------------------------------##
# AMX dialect.
##---------------------------------------------------------------------------##

td_library(
    name = "AMXTdFiles",
    srcs = ["include/mlir/Dialect/AMX/AMX.td"],
    includes = ["include"],
    deps = [
        ":LLVMOpsTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "AMXIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=amx",
            ],
            "include/mlir/Dialect/AMX/AMXDialect.h.inc",
        ),
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/AMX/AMX.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/AMX/AMX.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "g3doc/Dialects/AMX/AMX.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/AMX/AMX.td",
    deps = [":AMXTdFiles"],
)

cc_library(
    name = "AMX",
    srcs = ["lib/Dialect/AMX/IR/AMXDialect.cpp"],
    hdrs = ["include/mlir/Dialect/AMX/AMXDialect.h"],
    includes = ["include"],
    deps = [
        ":AMXIncGen",
        ":IR",
        ":LLVMDialect",
        ":SideEffectInterfaces",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "AMXTransforms",
    srcs = glob(["lib/Dialect/AMX/Transforms/*.cpp"]),
    hdrs = ["include/mlir/Dialect/AMX/Transforms.h"],
    includes = ["include"],
    deps = [
        ":AMX",
        ":IR",
        ":LLVMDialect",
        ":StandardOps",
        ":StandardToLLVM",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "AMXConversionIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-llvmir-conversions"],
            "include/mlir/Dialect/AMX/AMXConversions.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/AMX/AMX.td",
    deps = [":AMXTdFiles"],
)

##---------------------------------------------------------------------------##
# X86Vector dialect.
##---------------------------------------------------------------------------##

td_library(
    name = "X86VectorTdFiles",
    srcs = ["include/mlir/Dialect/X86Vector/X86Vector.td"],
    includes = ["include"],
    deps = [
        ":LLVMOpsTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "X86VectorIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=x86vector",
            ],
            "include/mlir/Dialect/X86Vector/X86VectorDialect.h.inc",
        ),
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/X86Vector/X86Vector.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/X86Vector/X86Vector.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "g3doc/Dialects/X86Vector/X86Vector.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/X86Vector/X86Vector.td",
    deps = [":X86VectorTdFiles"],
)

cc_library(
    name = "X86Vector",
    srcs = ["lib/Dialect/X86Vector/IR/X86VectorDialect.cpp"],
    hdrs = ["include/mlir/Dialect/X86Vector/X86VectorDialect.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":SideEffectInterfaces",
        ":X86VectorIncGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "X86VectorTransforms",
    srcs = glob(["lib/Dialect/X86Vector/Transforms/*.cpp"]),
    hdrs = ["include/mlir/Dialect/X86Vector/Transforms.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":StandardOps",
        ":StandardToLLVM",
        ":X86Vector",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "X86VectorConversionIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-llvmir-conversions"],
            "include/mlir/Dialect/X86Vector/X86VectorConversions.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/X86Vector/X86Vector.td",
    deps = [":X86VectorTdFiles"],
)

##---------------------------------------------------------------------------##
# SCF dialect.
##---------------------------------------------------------------------------##

td_library(
    name = "SCFTdFiles",
    srcs = ["include/mlir/Dialect/SCF/SCFOps.td"],
    includes = ["include"],
    deps = [
        ":ControlFlowInterfacesTdFiles",
        ":LoopLikeInterfaceTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "SCFIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/SCF/SCFOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/SCF/SCFOps.cpp.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "include/mlir/Dialect/SCF/SCFOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SCF/SCFOps.td",
    deps = [":SCFTdFiles"],
)

gentbl_cc_library(
    name = "SCFPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=SCF",
            ],
            "include/mlir/Dialect/SCF/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SCF/Passes.td",
    deps = [":PassBaseTdFiles"],
)

cc_library(
    name = "SCFTransforms",
    srcs = glob([
        "lib/Dialect/SCF/Transforms/*.cpp",
        "lib/Dialect/SCF/Transforms/*.h",
    ]),
    hdrs = ["include/mlir/Dialect/SCF/Passes.h"],
    includes = ["include"],
    deps = [
        ":Affine",
        ":IR",
        ":MemRefDialect",
        ":Pass",
        ":SCFDialect",
        ":SCFPassIncGen",
        ":StandardOps",
        ":Transforms",
        "@llvm-project//llvm:Support",
    ],
)

##---------------------------------------------------------------------------##
# SparseTensor dialect.
##---------------------------------------------------------------------------##

td_library(
    name = "SparseTensorTdFiles",
    srcs = [
        "include/mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.td",
        "include/mlir/Dialect/SparseTensor/IR/SparseTensorBase.td",
        "include/mlir/Dialect/SparseTensor/IR/SparseTensorOps.td",
    ],
    includes = ["include"],
    deps = [
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "SparseTensorAttrDefsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["--gen-attrdef-decls"],
            "include/mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.h.inc",
        ),
        (
            ["--gen-attrdef-defs"],
            "include/mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.td",
    deps = [":SparseTensorTdFiles"],
)

gentbl_cc_library(
    name = "SparseTensorOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=sparse_tensor",
            ],
            "include/mlir/Dialect/SparseTensor/IR/SparseTensorOpsDialect.h.inc",
        ),
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/SparseTensor/IR/SparseTensorOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/SparseTensor/IR/SparseTensorOps.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "g3doc/Dialects/SparseTensor/SparseTensor.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SparseTensor/IR/SparseTensorOps.td",
    deps = [":SparseTensorTdFiles"],
)

gentbl_cc_library(
    name = "SparseTensorPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=SparseTensor",
            ],
            "include/mlir/Dialect/SparseTensor/Transforms/Passes.h.inc",
        ),
        (
            [
                "-gen-pass-capi-header",
                "--prefix=SparseTensor",
            ],
            "include/mlir/Dialect/SparseTensor/Transforms/Passes.capi.h.inc",
        ),
        (
            [
                "-gen-pass-capi-impl",
                "--prefix=SparseTensor",
            ],
            "include/mlir/Dialect/SparseTensor/Transforms/Passes.capi.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SparseTensor/Transforms/Passes.td",
    deps = [":PassBaseTdFiles"],
)

cc_library(
    name = "SparseTensor",
    srcs = glob(["lib/Dialect/SparseTensor/IR/*.cpp"]),
    hdrs = ["include/mlir/Dialect/SparseTensor/IR/SparseTensor.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":SideEffectInterfaces",
        ":SparseTensorAttrDefsIncGen",
        ":SparseTensorOpsIncGen",
        ":StandardOps",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "SparseTensorTransforms",
    srcs = glob(["lib/Dialect/SparseTensor/Transforms/*.cpp"]),
    hdrs = ["include/mlir/Dialect/SparseTensor/Transforms/Passes.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":LinalgOps",
        ":LinalgTransforms",
        ":MemRefDialect",
        ":Pass",
        ":SCFDialect",
        ":SparseTensor",
        ":SparseTensorPassIncGen",
        ":StandardOps",
        ":StandardOpsTransforms",
        ":TensorDialect",
        ":Transforms",
        ":VectorOps",
        "@llvm-project//llvm:Support",
    ],
)

td_library(
    name = "StdOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/StandardOps/IR/Ops.td",
        "include/mlir/Dialect/StandardOps/IR/StandardOpsBase.td",
    ],
    includes = ["include"],
    deps = [
        ":CallInterfacesTdFiles",
        ":CastInterfacesTdFiles",
        ":ControlFlowInterfacesTdFiles",
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
        ":VectorInterfacesTdFiles",
        ":ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "StandardOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/StandardOps/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/StandardOps/IR/Ops.cpp.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "include/mlir/Dialect/StandardOps/IR/OpsDialect.h.inc",
        ),
        (
            ["-gen-enum-decls"],
            "include/mlir/Dialect/StandardOps/IR/OpsEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/mlir/Dialect/StandardOps/IR/OpsEnums.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/StandardOps/IR/Ops.td",
    deps = [":StdOpsTdFiles"],
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
        "@llvm-project//llvm:Support",
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
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "Affine",
    srcs = glob(
        [
            "lib/Dialect/Affine/IR/*.cpp",
            "lib/Dialect/Affine/IR/*.h",
        ],
    ) + ["include/mlir/Transforms/InliningUtils.h"],
    hdrs = glob([
        "include/mlir/Dialect/Affine/IR/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":AffineMemoryOpInterfacesIncGen",
        ":AffineOpsIncGen",
        ":IR",
        ":LoopLikeInterface",
        ":MemRefDialect",
        ":SideEffectInterfaces",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "Async",
    srcs = glob([
        "lib/Dialect/Async/IR/*.cpp",
    ]),
    hdrs = glob([
        "include/mlir/Dialect/Async/IR/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":AsyncOpsIncGen",
        ":ControlFlowInterfaces",
        ":Dialect",
        ":IR",
        ":SideEffectInterfaces",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "AsyncTransforms",
    srcs = glob([
        "lib/Dialect/Async/Transforms/*.cpp",
        "lib/Dialect/Async/Transforms/*.h",
    ]),
    hdrs = ["include/mlir/Dialect/Async/Passes.h"],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":Async",
        ":AsyncPassIncGen",
        ":IR",
        ":Pass",
        ":SCFDialect",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":Transforms",
        ":TransformsPassIncGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "AffineUtils",
    srcs = glob(
        [
            "lib/Dialect/Affine/Utils/*.cpp",
            "lib/Dialect/Affine/Utils/*.h",
        ],
    ),
    hdrs = ["include/mlir/Dialect/Affine/Utils.h"],
    includes = ["include"],
    deps = [
        ":Affine",
        ":Analysis",
        ":IR",
        ":Support",
        ":TransformUtils",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "AffinePassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Affine",
            ],
            "include/mlir/Dialect/Affine/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Affine/Passes.td",
    deps = [":PassBaseTdFiles"],
)

cc_library(
    name = "AffineTransforms",
    srcs = glob([
        "lib/Dialect/Affine/Transforms/*.cpp",
        "lib/Dialect/Affine/Transforms/*.h",
    ]),
    hdrs = ["include/mlir/Dialect/Affine/Passes.h"],
    includes = ["include"],
    deps = [
        ":Affine",
        ":AffinePassIncGen",
        ":AffineUtils",
        ":Analysis",
        ":IR",
        ":MemRefDialect",
        ":Pass",
        ":SCFDialect",
        ":StandardOps",
        ":Support",
        ":Transforms",
        ":VectorOps",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "ConversionPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Conversion",
            ],
            "include/mlir/Conversion/Passes.h.inc",
        ),
        (
            [
                "-gen-pass-capi-header",
                "--prefix=Conversion",
            ],
            "include/mlir/Conversion/Passes.capi.h.inc",
        ),
        (
            [
                "-gen-pass-capi-impl",
                "--prefix=Conversion",
            ],
            "include/mlir/Conversion/Passes.capi.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Conversion/Passes.td",
    deps = [":PassBaseTdFiles"],
)

cc_library(
    name = "ConversionPasses",
    hdrs = ["include/mlir/Conversion/Passes.h"],
    includes = ["include"],
    deps = [
        ":AffineToStandard",
        ":AsyncToLLVM",
        ":ComplexToLLVM",
        ":ComplexToStandard",
        ":ConversionPassIncGen",
        ":GPUToGPURuntimeTransforms",
        ":GPUToNVVMTransforms",
        ":GPUToROCDLTransforms",
        ":GPUToSPIRV",
        ":GPUToVulkanTransforms",
        ":LinalgToLLVM",
        ":LinalgToSPIRV",
        ":LinalgToStandard",
        ":MathToLibm",
        ":OpenACCToLLVM",
        ":OpenMPToLLVM",
        ":PDLToPDLInterp",
        ":SCFToGPUPass",
        ":SCFToOpenMP",
        ":SCFToSPIRV",
        ":SCFToStandard",
        ":SPIRVToLLVM",
        ":ShapeToStandard",
        ":StandardToLLVM",
        ":StandardToSPIRV",
        ":TosaToLinalg",
        ":TosaToSCF",
        ":TosaToStandard",
        ":VectorToLLVM",
        ":VectorToROCDL",
        ":VectorToSCF",
        ":VectorToSPIRV",
    ],
)

cc_library(
    name = "AsyncToLLVM",
    srcs = glob([
        "lib/Conversion/AsyncToLLVM/*.cpp",
        "lib/Conversion/AsyncToLLVM/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob(["include/mlir/Conversion/AsyncToLLVM/*.h"]),
    includes = ["include"],
    deps = [
        ":Async",
        ":ConversionPassIncGen",
        ":IR",
        ":LLVMDialect",
        ":Pass",
        ":StandardOps",
        ":StandardOpsTransforms",
        ":StandardToLLVM",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "AffineToStandard",
    srcs = glob([
        "lib/Conversion/AffineToStandard/*.cpp",
        "lib/Conversion/AffineToStandard/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob(["include/mlir/Conversion/AffineToStandard/*.h"]),
    includes = ["include"],
    deps = [
        ":Affine",
        ":ConversionPassIncGen",
        ":IR",
        ":MemRefDialect",
        ":Pass",
        ":SCFDialect",
        ":StandardOps",
        ":Support",
        ":Transforms",
        ":VectorOps",
    ],
)

alias(
    name = "AffineToStandardTransforms",
    actual = "AffineToStandard",
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
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "SCFDialect",
    srcs = glob(
        [
            "lib/Dialect/SCF/*.cpp",
            "lib/Dialect/SCF/*.h",
        ],
    ),
    hdrs = glob([
        "include/mlir/Dialect/SCF/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":ControlFlowInterfaces",
        ":IR",
        ":LoopLikeInterface",
        ":MemRefDialect",
        ":Pass",
        ":SCFIncGen",
        ":SCFPassIncGen",
        ":StandardOps",
        ":Support",
        ":TensorDialect",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "LinalgInterfaces",
    srcs = ["lib/Dialect/Linalg/IR/LinalgInterfaces.cpp"],
    hdrs = ["include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h"],
    includes = ["include"],
    deps = [
        ":Affine",
        ":DialectUtils",
        ":IR",
        ":LinalgInterfacesIncGen",
        ":LinalgStructuredOpsIncGen",
        ":MemRefDialect",
        ":ViewLikeInterface",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "DataLayoutInterfaces",
    srcs = ["lib/Interfaces/DataLayoutInterfaces.cpp"],
    hdrs = ["include/mlir/Interfaces/DataLayoutInterfaces.h"],
    includes = ["include"],
    deps = [
        ":DataLayoutInterfacesIncGen",
        ":IR",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "LoopLikeInterface",
    srcs = ["lib/Interfaces/LoopLikeInterface.cpp"],
    hdrs = ["include/mlir/Interfaces/LoopLikeInterface.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":LoopLikeInterfaceIncGen",
    ],
)

cc_library(
    name = "VectorInterfaces",
    srcs = ["lib/Interfaces/VectorInterfaces.cpp"],
    hdrs = ["include/mlir/Interfaces/VectorInterfaces.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":VectorInterfacesIncGen",
    ],
)

cc_library(
    name = "ViewLikeInterface",
    srcs = ["lib/Interfaces/ViewLikeInterface.cpp"],
    hdrs = ["include/mlir/Interfaces/ViewLikeInterface.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":ViewLikeInterfaceIncGen",
    ],
)

cc_library(
    name = "CopyOpInterface",
    srcs = ["lib/Interfaces/CopyOpInterface.cpp"],
    hdrs = ["include/mlir/Interfaces/CopyOpInterface.h"],
    includes = ["include"],
    deps = [
        ":CopyOpInterfaceIncGen",
        ":IR",
    ],
)

td_library(
    name = "ShapeOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Shape/IR/ShapeBase.td",
        "include/mlir/Dialect/Shape/IR/ShapeOps.td",
    ],
    includes = ["include"],
    deps = [
        ":ControlFlowInterfacesTdFiles",
        ":InferTypeOpInterfaceTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "ShapeOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/Shape/IR/ShapeOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/Shape/IR/ShapeOps.cpp.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "include/mlir/Dialect/Shape/IR/ShapeOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Shape/IR/ShapeOps.td",
    deps = [":ShapeOpsTdFiles"],
)

gentbl_cc_library(
    name = "MLIRShapeCanonicalizationIncGen",
    strip_include_prefix = "include/mlir/Dialect/Shape/IR",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "include/mlir/Dialect/Shape/IR/ShapeCanonicalization.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "lib/Dialect/Shape/IR/ShapeCanonicalization.td",
    deps = [
        ":ShapeOpsTdFiles",
        ":StdOpsTdFiles",
        ":TensorOpsTdFiles",
    ],
)

cc_library(
    name = "Shape",
    srcs = glob(["lib/Dialect/Shape/IR/*.cpp"]),
    hdrs = ["include/mlir/Dialect/Shape/IR/Shape.h"],
    includes = ["include"],
    deps = [
        ":CallOpInterfaces",
        ":CommonFolders",
        ":ControlFlowInterfaces",
        ":Dialect",
        ":IR",
        ":InferTypeOpInterface",
        ":MLIRShapeCanonicalizationIncGen",
        ":ShapeOpsIncGen",
        ":SideEffectInterfaces",
        ":StandardOps",
        ":Support",
        ":TensorDialect",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "ShapeToStandardGen",
    strip_include_prefix = "lib/Conversion/ShapeToStandard",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "lib/Conversion/ShapeToStandard/ShapeToStandard.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "lib/Conversion/ShapeToStandard/ShapeToStandard.td",
    deps = [":ShapeOpsTdFiles"],
)

cc_library(
    name = "ShapeToStandard",
    srcs = glob([
        "lib/Conversion/ShapeToStandard/*.cpp",
        "lib/Conversion/ShapeToStandard/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = ["include/mlir/Conversion/ShapeToStandard/ShapeToStandard.h"],
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":MemRefDialect",
        ":Pass",
        ":SCFDialect",
        ":Shape",
        ":ShapeToStandardGen",
        ":StandardOps",
        ":Support",
        ":TensorDialect",
        ":Transforms",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "ShapeTransformsPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [(
        [
            "-gen-pass-decls",
            "-name=Shape",
        ],
        "include/mlir/Dialect/Shape/Transforms/Passes.h.inc",
    )],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Shape/Transforms/Passes.td",
    deps = [":PassBaseTdFiles"],
)

cc_library(
    name = "ShapeTransforms",
    srcs = glob([
        "lib/Dialect/Shape/Transforms/*.cpp",
        "lib/Dialect/Shape/Transforms/*.h",
    ]),
    hdrs = ["include/mlir/Dialect/Shape/Transforms/Passes.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":MemRefDialect",
        ":Pass",
        ":Shape",
        ":ShapeTransformsPassIncGen",
        ":StandardOps",
        ":Transforms",
    ],
)

cc_library(
    name = "StandardOps",
    srcs = glob(
        [
            "lib/Dialect/StandardOps/IR/*.cpp",
            "lib/Dialect/StandardOps/IR/*.h",
            "lib/Dialect/StandardOps/EDSC/*.cpp",
            "lib/Dialect/StandardOps/Utils/*.cpp",
        ],
    ) +
    # TODO: Stop depending on MemRefOps.
    [
        "include/mlir/Dialect/MemRef/IR/MemRef.h",
    ],
    hdrs = glob([
        "include/mlir/Dialect/StandardOps/IR/*.h",
        "include/mlir/Dialect/StandardOps/EDSC/*.h",
        "include/mlir/Dialect/StandardOps/Utils/*.h",
    ]) + ["include/mlir/Transforms/InliningUtils.h"],
    includes = ["include"],
    deps = [
        ":CallOpInterfaces",
        ":CastOpInterfaces",
        ":CommonFolders",
        ":ControlFlowInterfaces",
        ":CopyOpInterface",  # TODO: Stop depending on MemRefOps.
        ":EDSC",
        ":IR",
        ":MemRefBaseIncGen",  # TODO: Stop depending on MemRefOps.
        ":MemRefOpsIncGen",  # TODO: Stop depending on MemRefOps.
        ":SideEffectInterfaces",
        ":StandardOpsIncGen",
        ":Support",
        ":TensorDialect",
        ":VectorInterfaces",
        ":ViewLikeInterface",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "StandardOpsTransformsPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [(
        [
            "-gen-pass-decls",
            "-name=Standard",
        ],
        "include/mlir/Dialect/StandardOps/Transforms/Passes.h.inc",
    )],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/StandardOps/Transforms/Passes.td",
    deps = [":PassBaseTdFiles"],
)

cc_library(
    name = "StandardOpsTransforms",
    srcs = glob([
        "lib/Dialect/StandardOps/Transforms/*.cpp",
        "lib/Dialect/StandardOps/Transforms/*.h",
    ]),
    hdrs = glob(["include/mlir/Dialect/StandardOps/Transforms/*.h"]),
    includes = ["include"],
    deps = [
        ":Affine",
        ":IR",
        ":MemRefDialect",
        ":Pass",
        ":SCFDialect",
        ":StandardOps",
        ":StandardOpsTransformsPassIncGen",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "VectorOps",
    srcs = glob(
        [
            "lib/Dialect/Vector/*.cpp",
            "lib/Dialect/Vector/*.h",
        ],
    ),
    hdrs = glob([
        "include/mlir/Dialect/Vector/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":Affine",
        ":AffineUtils",
        ":Analysis",
        ":DataLayoutInterfaces",
        ":DialectUtils",
        ":IR",
        ":LinalgOps",
        ":MemRefDialect",
        ":SCFDialect",
        ":SideEffectInterfaces",
        ":StandardOps",
        ":Support",
        ":TensorDialect",
        ":VectorInterfaces",
        ":VectorOpsIncGen",
        ":ViewLikeInterface",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "Support",
    srcs = glob(
        [
            "lib/Support/*.cpp",
            "lib/Support/*.h",
        ],
        exclude = [
            # TODO(jpienaar): Move this out, else Support depends on Analysis/
            "lib/Support/MlirOptMain.cpp",
        ],
    ),
    hdrs = glob(
        ["include/mlir/Support/*.h"],
        exclude = ["include/mlir/Support/MlirOptMain.h"],
    ),
    includes = ["include"],
    deps = ["@llvm-project//llvm:Support"],
)

cc_library(
    name = "MlirLspServerLib",
    srcs = glob(
        [
            "lib/Tools/mlir-lsp-server/*.cpp",
            "lib/Tools/mlir-lsp-server/*.h",
            "lib/Tools/mlir-lsp-server/lsp/*.cpp",
            "lib/Tools/mlir-lsp-server/lsp/*.h",
        ],
    ),
    hdrs = glob(
        ["include/mlir/Tools/mlir-lsp-server/*.h"],
    ),
    includes = ["include"],
    deps = [
        ":IR",
        ":Parser",
        ":Support",
        "@llvm-project//llvm:Support",
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
        "include/mlir/Parser/*.h",
    ]) + [
        "include/mlir/Parser.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":ParserTokenKinds",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "LLVMDialectInterfaceIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Dialect/LLVMIR/LLVMOpsInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Dialect/LLVMIR/LLVMOpsInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/LLVMOpsInterfaces.td",
    deps = [":LLVMOpsTdFiles"],
)

gentbl_cc_library(
    name = "LLVMDialectAttributesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["--gen-attrdef-decls"],
            "include/mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.h.inc",
        ),
        (
            ["--gen-attrdef-defs"],
            "include/mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/LLVMAttrDefs.td",
    deps = [":LLVMOpsTdFiles"],
)

cc_library(
    name = "LLVMDialect",
    srcs = glob(
        [
            "lib/Dialect/LLVMIR/IR/*.cpp",
            "lib/Dialect/LLVMIR/IR/*.h",
        ],
        exclude = [
            "lib/Dialect/LLVMIR/IR/*AMX*.cpp",
            "lib/Dialect/LLVMIR/IR/*AMX*.h",
            "lib/Dialect/LLVMIR/IR/*ArmSVE*.cpp",
            "lib/Dialect/LLVMIR/IR/*ArmSVE*.h",
            "lib/Dialect/LLVMIR/IR/NVVM*.cpp",
            "lib/Dialect/LLVMIR/IR/NVVM*.h",
            "lib/Dialect/LLVMIR/IR/ROCDL*.cpp",
            "lib/Dialect/LLVMIR/IR/ROCDL*.h",
            "lib/Dialect/LLVMIR/IR/*X86Vector*.cpp",
            "lib/Dialect/LLVMIR/IR/*X86Vector*.h",
        ],
    ),
    hdrs = glob(
        ["include/mlir/Dialect/LLVMIR/*.h"],
        exclude = [
            "include/mlir/Dialect/LLVMIR/*AMX*.h",
            "include/mlir/Dialect/LLVMIR/*ArmSVE*.h",
            "include/mlir/Dialect/LLVMIR/NVVM*.h",
            "include/mlir/Dialect/LLVMIR/ROCDL*.h",
            "include/mlir/Dialect/LLVMIR/*X86Vector*.h",
        ],
    ),
    includes = ["include"],
    deps = [
        ":ControlFlowInterfaces",
        ":IR",
        ":LLVMDialectAttributesIncGen",
        ":LLVMDialectInterfaceIncGen",
        ":LLVMOpsIncGen",
        ":SideEffectInterfaces",
        ":Support",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "LLVMPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=LLVM",
            ],
            "include/mlir/Dialect/LLVMIR/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/Transforms/Passes.td",
    deps = [":PassBaseTdFiles"],
)

cc_library(
    name = "LLVMIRTransforms",
    srcs = glob([
        "lib/Dialect/LLVMIR/Transforms/*.cpp",
        "lib/Dialect/LLVMIR/Transforms/*.h",
    ]),
    hdrs = glob(["include/mlir/Dialect/LLVMIR/Transforms/*.h"]),
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":LLVMPassIncGen",
        ":Pass",
    ],
)

td_library(
    name = "GPUOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/GPU/GPUBase.td",
        "include/mlir/Dialect/GPU/GPUOps.td",
    ],
    includes = ["include"],
    deps = [
        ":DLTIDialectTdFiles",
        ":DataLayoutInterfacesTdFiles",
        ":LLVMOpsTdFiles",
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "ParallelLoopMapperAttrGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-struct-attr-decls"],
            "include/mlir/Dialect/GPU/ParallelLoopMapperAttr.h.inc",
        ),
        (
            ["-gen-struct-attr-defs"],
            "include/mlir/Dialect/GPU/ParallelLoopMapperAttr.cpp.inc",
        ),
        (
            ["-gen-enum-decls"],
            "include/mlir/Dialect/GPU/ParallelLoopMapperEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/mlir/Dialect/GPU/ParallelLoopMapperEnums.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/GPU/ParallelLoopMapperAttr.td",
    deps = [":GPUOpsTdFiles"],
)

gentbl_cc_library(
    name = "GPUBaseIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=gpu",
            ],
            "include/mlir/Dialect/GPU/GPUOpsDialect.h.inc",
        ),
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Dialect/GPU/GPUOpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Dialect/GPU/GPUOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/GPU/GPUBase.td",
    deps = [":GPUOpsTdFiles"],
)

gentbl_cc_library(
    name = "GPUOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/GPU/GPUOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/GPU/GPUOps.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/GPU/GPUOps.td",
    deps = [
        ":DLTIDialectTdFiles",
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
    ),
    hdrs = ["include/mlir/Dialect/GPU/GPUDialect.h"],
    includes = ["include"],
    deps = [
        ":DLTIDialect",
        ":GPUBaseIncGen",
        ":GPUOpsIncGen",
        ":IR",
        ":LLVMDialect",
        ":SideEffectInterfaces",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "GPUPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=GPU",
            ],
            "include/mlir/Dialect/GPU/Passes.h.inc",
        ),
        (
            [
                "-gen-pass-capi-header",
                "--prefix=GPU",
            ],
            "include/mlir/Dialect/GPU/Passes.capi.h.inc",
        ),
        (
            [
                "-gen-pass-capi-impl",
                "--prefix=GPU",
            ],
            "include/mlir/Dialect/GPU/Passes.capi.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/GPU/Passes.td",
    deps = [":PassBaseTdFiles"],
)

cc_library(
    name = "GPUTransforms",
    srcs = glob(
        [
            "lib/Dialect/GPU/Transforms/*.cpp",
            "lib/Dialect/GPU/Transforms/*.h",
        ],
    ),
    hdrs = [
        "include/mlir/Dialect/GPU/MemoryPromotion.h",
        "include/mlir/Dialect/GPU/ParallelLoopMapper.h",
        "include/mlir/Dialect/GPU/Passes.h",
        "include/mlir/Dialect/GPU/Utils.h",
    ],
    defines = if_cuda_available(["MLIR_GPU_TO_CUBIN_PASS_ENABLE"]),
    includes = ["include"],
    deps = [
        ":Async",
        ":EDSC",
        ":GPUDialect",
        ":GPUPassIncGen",
        ":MemRefDialect",
        ":IR",
        ":ParallelLoopMapperAttrGen",
        ":Pass",
        ":SCFDialect",
        ":StandardOps",
        ":Support",
        ":Transforms",
        ":ToLLVMIRTranslation",
        ":LLVMToLLVMIRTranslation",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
    ] + if_cuda_available([
        # Dependencies for SerializeToCubin.cpp with
        # -DMLIR_GPU_TO_CUBIN_PASS_ENABLE
        ":NVVMToLLVMIRTranslation",
        "@llvm-project//llvm:NVPTXCodeGen",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:libcuda",
    ]),
)

td_library(
    name = "LLVMOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/LLVMIR/LLVMOpBase.td",
        "include/mlir/Dialect/LLVMIR/LLVMOps.td",
        "include/mlir/Dialect/LLVMIR/LLVMOpsInterfaces.td",
    ],
    includes = ["include"],
    deps = [
        ":ControlFlowInterfacesTdFiles",
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

cc_library(
    name = "GPUCommonTransforms",
    srcs = [
        "lib/Conversion/GPUCommon/GPUOpsLowering.cpp",
    ],
    hdrs = [
        "lib/Conversion/GPUCommon/GPUOpsLowering.h",
        "lib/Conversion/GPUCommon/IndexIntrinsicsOpLowering.h",
        "lib/Conversion/GPUCommon/OpToFuncCallLowering.h",
    ],
    deps = [
        ":GPUDialect",
        ":IR",
        ":LLVMDialect",
        ":StandardOps",
        ":StandardToLLVM",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "GPUToNVVMGen",
    strip_include_prefix = "lib/Conversion/GPUToNVVM",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "lib/Conversion/GPUToNVVM/GPUToNVVM.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "lib/Conversion/GPUToNVVM/GPUToNVVM.td",
    deps = [
        ":GPUOpsTdFiles",
        ":NVVMOpsTdFiles",
    ],
)

cc_library(
    name = "GPUToNVVMTransforms",
    srcs = glob([
        "lib/Conversion/GPUToNVVM/*.cpp",
        "lib/Conversion/GPUToNVVM/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/GPUToNVVM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":GPUCommonTransforms",
        ":GPUDialect",
        ":GPUToNVVMGen",
        ":GPUTransforms",
        ":IR",
        ":MathDialect",
        ":MemRefDialect",
        ":NVVMDialect",
        ":Pass",
        ":StandardToLLVM",
        ":Transforms",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "VectorToROCDL",
    srcs = [
        "lib/Conversion/PassDetail.h",
        "lib/Conversion/VectorToROCDL/VectorToROCDL.cpp",
    ],
    hdrs = ["include/mlir/Conversion/VectorToROCDL/VectorToROCDL.h"],
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":GPUDialect",
        ":LLVMDialect",
        ":Pass",
        ":ROCDLDialect",
        ":StandardOps",
        ":StandardToLLVM",
        ":Transforms",
        ":VectorOps",
    ],
)

cc_library(
    name = "VectorToSPIRV",
    srcs = glob([
        "lib/Conversion/VectorToSPIRV/*.cpp",
        "lib/Conversion/VectorToSPIRV/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/VectorToSPIRV/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":Pass",
        ":SPIRVConversion",
        ":SPIRVDialect",
        ":Transforms",
        ":VectorOps",
    ],
)

gentbl_cc_library(
    name = "GPUToROCDLTGen",
    strip_include_prefix = "lib/Conversion/GPUToROCDL",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "lib/Conversion/GPUToROCDL/GPUToROCDL.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "lib/Conversion/GPUToROCDL/GPUToROCDL.td",
    deps = [
        ":GPUOpsTdFiles",
        ":ROCDLOpsTdFiles",
    ],
)

cc_library(
    name = "GPUToROCDLTransforms",
    srcs = [
        "lib/Conversion/GPUToROCDL/LowerGpuOpsToROCDLOps.cpp",
        "lib/Conversion/PassDetail.h",
    ],
    hdrs = ["include/mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"],
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":GPUCommonTransforms",
        ":GPUDialect",
        ":GPUToROCDLTGen",
        ":GPUTransforms",
        ":MathDialect",
        ":Pass",
        ":ROCDLDialect",
        ":StandardToLLVM",
        ":Transforms",
        ":VectorOps",
        ":VectorToLLVM",
        ":VectorToROCDL",
        ":VectorToSCF",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "GPUToVulkanTransforms",
    srcs = [
        "lib/Conversion/GPUToVulkan/ConvertGPULaunchFuncToVulkanLaunchFunc.cpp",
        "lib/Conversion/GPUToVulkan/ConvertLaunchFuncToVulkanCalls.cpp",
        "lib/Conversion/PassDetail.h",
    ],
    hdrs = ["include/mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"],
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":GPUDialect",
        ":IR",
        ":LLVMDialect",
        ":Pass",
        ":SPIRVDialect",
        ":SPIRVSerialization",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "GPUToGPURuntimeTransforms",
    srcs = [
        "lib/Conversion/GPUCommon/GPUToLLVMConversion.cpp",
        "lib/Conversion/PassDetail.h",
    ],
    hdrs = ["include/mlir/Conversion/GPUCommon/GPUCommonPass.h"],
    includes = ["include"],
    deps = [
        ":Async",
        ":AsyncToLLVM",
        ":ConversionPassIncGen",
        ":GPUDialect",
        ":GPUTransforms",
        ":IR",
        ":LLVMDialect",
        ":NVVMToLLVMIRTranslation",
        ":Pass",
        ":StandardToLLVM",
        ":Support",
        ":VectorToLLVM",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "GPUToSPIRVIncGen",
    strip_include_prefix = "lib/Conversion/GPUToSPIRV",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "lib/Conversion/GPUToSPIRV/GPUToSPIRV.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "lib/Conversion/GPUToSPIRV/GPUToSPIRV.td",
    deps = [
        ":GPUOpsTdFiles",
        ":SPIRVOpsTdFiles",
    ],
)

cc_library(
    name = "GPUToSPIRV",
    srcs = glob([
        "lib/Conversion/GPUToSPIRV/*.cpp",
        "lib/Conversion/GPUToSPIRV/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/GPUToSPIRV/*.h",
    ]),
    includes = [
        "include",
        "lib/Conversions/GPUToSPIRV",
    ],
    deps = [
        ":ConversionPassIncGen",
        ":GPUDialect",
        ":GPUToSPIRVIncGen",
        ":IR",
        ":Pass",
        ":SCFDialect",
        ":SCFToSPIRV",
        ":SPIRVConversion",
        ":SPIRVDialect",
        ":StandardToSPIRV",
        ":Support",
        ":Transforms",
        ":VectorToSPIRV",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "PDLToPDLInterp",
    srcs = glob([
        "lib/Conversion/PDLToPDLInterp/*.cpp",
        "lib/Conversion/PDLToPDLInterp/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = ["include/mlir/Conversion/PDLToPDLInterp/PDLToPDLInterp.h"],
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":InferTypeOpInterface",
        ":PDLDialect",
        ":PDLInterpDialect",
        ":Pass",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "SPIRVToLLVM",
    srcs = glob([
        "lib/Conversion/SPIRVToLLVM/*.cpp",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/SPIRVToLLVM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":GPUDialect",
        ":IR",
        ":LLVMDialect",
        ":Pass",
        ":SPIRVDialect",
        ":SPIRVUtils",
        ":StandardOps",
        ":StandardToLLVM",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "LLVMOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/LLVMIR/LLVMOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/LLVMIR/LLVMOps.cpp.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "include/mlir/Dialect/LLVMIR/LLVMOpsDialect.h.inc",
        ),
        (
            ["-gen-enum-decls"],
            "include/mlir/Dialect/LLVMIR/LLVMOpsEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/mlir/Dialect/LLVMIR/LLVMOpsEnums.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/LLVMOps.td",
    deps = [":LLVMOpsTdFiles"],
)

gentbl_cc_library(
    name = "LLVMConversionIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-llvmir-conversions"],
            "include/mlir/Dialect/LLVMIR/LLVMConversions.inc",
        ),
        (
            ["-gen-enum-to-llvmir-conversions"],
            "include/mlir/Dialect/LLVMIR/LLVMConversionEnumsToLLVM.inc",
        ),
        (
            ["-gen-enum-from-llvmir-conversions"],
            "include/mlir/Dialect/LLVMIR/LLVMConversionEnumsFromLLVM.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/LLVMOps.td",
    deps = [":LLVMOpsTdFiles"],
)

cc_library(
    name = "NVVMDialect",
    srcs = ["lib/Dialect/LLVMIR/IR/NVVMDialect.cpp"],
    hdrs = ["include/mlir/Dialect/LLVMIR/NVVMDialect.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":NVVMOpsIncGen",
        ":SideEffectInterfaces",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

td_library(
    name = "NVVMOpsTdFiles",
    srcs = ["include/mlir/Dialect/LLVMIR/NVVMOps.td"],
    includes = ["include"],
    deps = [
        ":LLVMOpsTdFiles",
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "NVVMOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/LLVMIR/NVVMOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/LLVMIR/NVVMOps.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=nvvm",
            ],
            "include/mlir/Dialect/LLVMIR/NVVMOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/NVVMOps.td",
    deps = [":NVVMOpsTdFiles"],
)

gentbl_cc_library(
    name = "NVVMConversionIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-llvmir-conversions"],
            "include/mlir/Dialect/LLVMIR/NVVMConversions.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/NVVMOps.td",
    deps = [":NVVMOpsTdFiles"],
)

cc_library(
    name = "ROCDLDialect",
    srcs = ["lib/Dialect/LLVMIR/IR/ROCDLDialect.cpp"],
    hdrs = ["include/mlir/Dialect/LLVMIR/ROCDLDialect.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":ROCDLOpsIncGen",
        ":SideEffectInterfaces",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

td_library(
    name = "ROCDLOpsTdFiles",
    srcs = ["include/mlir/Dialect/LLVMIR/ROCDLOps.td"],
    includes = ["include"],
    deps = [
        ":LLVMOpsTdFiles",
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "ROCDLOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/LLVMIR/ROCDLOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/LLVMIR/ROCDLOps.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=rocdl",
            ],
            "include/mlir/Dialect/LLVMIR/ROCDLOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/ROCDLOps.td",
    deps = [":ROCDLOpsTdFiles"],
)

gentbl_cc_library(
    name = "ROCDLConversionIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-llvmir-conversions"],
            "include/mlir/Dialect/LLVMIR/ROCDLConversions.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/LLVMIR/ROCDLOps.td",
    deps = [":ROCDLOpsTdFiles"],
)

cc_library(
    name = "PDLDialect",
    srcs = glob([
        "lib/Dialect/PDL/IR/*.cpp",
        "lib/Dialect/PDL/IR/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Dialect/PDL/IR/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":IR",
        ":InferTypeOpInterface",
        ":PDLOpsIncGen",
        ":PDLTypesIncGen",
        ":SideEffects",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

td_library(
    name = "PDLDialectTdFiles",
    srcs = [
        "include/mlir/Dialect/PDL/IR/PDLDialect.td",
        "include/mlir/Dialect/PDL/IR/PDLOps.td",
        "include/mlir/Dialect/PDL/IR/PDLTypes.td",
    ],
    deps = [
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "PDLOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/PDL/IR/PDLOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/PDL/IR/PDLOps.cpp.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "include/mlir/Dialect/PDL/IR/PDLOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/PDL/IR/PDLOps.td",
    deps = [":PDLDialectTdFiles"],
)

gentbl_cc_library(
    name = "PDLTypesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "include/mlir/Dialect/PDL/IR/PDLOpsTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "include/mlir/Dialect/PDL/IR/PDLOpsTypes.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/PDL/IR/PDLTypes.td",
    deps = [":PDLDialectTdFiles"],
)

cc_library(
    name = "PDLInterpDialect",
    srcs = glob([
        "lib/Dialect/PDLInterp/IR/*.cpp",
        "lib/Dialect/PDLInterp/IR/*.h",
    ]),
    hdrs = glob([
        "include/mlir/Dialect/PDLInterp/IR/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":IR",
        ":InferTypeOpInterface",
        ":PDLDialect",
        ":PDLInterpOpsIncGen",
        ":SideEffects",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

td_library(
    name = "PDLInterpOpsTdFiles",
    srcs = ["include/mlir/Dialect/PDLInterp/IR/PDLInterpOps.td"],
    includes = ["include"],
    deps = [
        ":OpBaseTdFiles",
        ":PDLDialectTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "PDLInterpOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/PDLInterp/IR/PDLInterpOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/PDLInterp/IR/PDLInterpOps.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=pdl_interp",
            ],
            "include/mlir/Dialect/PDLInterp/IR/PDLInterpOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/PDLInterp/IR/PDLInterpOps.td",
    deps = [":PDLInterpOpsTdFiles"],
)

td_library(
    name = "SPIRVOpsTdFiles",
    srcs = glob(["include/mlir/Dialect/SPIRV/IR/*.td"]),
    includes = ["include"],
    deps = [
        ":CallInterfacesTdFiles",
        ":ControlFlowInterfacesTdFiles",
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "SPIRVOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/SPIRV/IR/SPIRVOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/SPIRV/IR/SPIRVOps.cpp.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "include/mlir/Dialect/SPIRV/IR/SPIRVOpsDialect.h.inc",
        ),
        (
            ["-gen-op-doc"],
            "g3doc/Dialects/SPIRV/SPIRVOps.md",
        ),
        (
            ["-gen-enum-decls"],
            "include/mlir/Dialect/SPIRV/IR/SPIRVEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/mlir/Dialect/SPIRV/IR/SPIRVEnums.cpp.inc",
        ),
        (
            ["-gen-spirv-enum-avail-decls"],
            "include/mlir/Dialect/SPIRV/IR/SPIRVEnumAvailability.h.inc",
        ),
        (
            ["-gen-spirv-enum-avail-defs"],
            "include/mlir/Dialect/SPIRV/IR/SPIRVEnumAvailability.cpp.inc",
        ),
        (
            ["-gen-spirv-capability-implication"],
            "include/mlir/Dialect/SPIRV/IR/SPIRVCapabilityImplication.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/IR/SPIRVOps.td",
    deps = [":SPIRVOpsTdFiles"],
)

gentbl_cc_library(
    name = "SPIRVCanonicalizationIncGen",
    strip_include_prefix = "lib/Dialect/SPIRV/IR",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "lib/Dialect/SPIRV/IR/SPIRVCanonicalization.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "lib/Dialect/SPIRV/IR/SPIRVCanonicalization.td",
    deps = [":SPIRVOpsTdFiles"],
)

gentbl_cc_library(
    name = "SPIRVAvailabilityIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-avail-interface-decls"],
            "include/mlir/Dialect/SPIRV/IR/SPIRVAvailability.h.inc",
        ),
        (
            ["-gen-avail-interface-defs"],
            "include/mlir/Dialect/SPIRV/IR/SPIRVAvailability.cpp.inc",
        ),
        (
            ["-gen-spirv-avail-impls"],
            "include/mlir/Dialect/SPIRV/IR/SPIRVOpAvailabilityImpl.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/IR/SPIRVOps.td",
    deps = [":SPIRVOpsTdFiles"],
)

gentbl_cc_library(
    name = "SPIRVTargetAndABIStructGen",
    tbl_outs = [
        (
            ["-gen-struct-attr-decls"],
            "include/mlir/Dialect/SPIRV/IR/TargetAndABI.h.inc",
        ),
        (
            ["-gen-struct-attr-defs"],
            "include/mlir/Dialect/SPIRV/IR/TargetAndABI.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/IR/TargetAndABI.td",
    deps = [":SPIRVOpsTdFiles"],
)

gentbl_cc_library(
    name = "SPIRVAttrUtilsGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-spirv-attr-utils"],
            "include/mlir/Dialect/SPIRV/IR/SPIRVAttrUtils.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/IR/SPIRVBase.td",
    deps = [":SPIRVOpsTdFiles"],
)

gentbl_cc_library(
    name = "SPIRVSerializationGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-spirv-serialization"],
            "include/mlir/Dialect/SPIRV/IR/SPIRVSerialization.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/IR/SPIRVOps.td",
    deps = [":SPIRVOpsTdFiles"],
)

cc_library(
    name = "SPIRVDialect",
    srcs = glob([
        "lib/Dialect/SPIRV/IR/*.cpp",
        "lib/Dialect/SPIRV/IR/*.h",
    ]) + ["include/mlir/Transforms/InliningUtils.h"],
    hdrs = glob([
        "include/mlir/Dialect/SPIRV/IR/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":CommonFolders",
        ":ControlFlowInterfaces",
        ":IR",
        ":Parser",
        ":Pass",
        ":SPIRVAttrUtilsGen",
        ":SPIRVAvailabilityIncGen",
        ":SPIRVCanonicalizationIncGen",
        ":SPIRVOpsIncGen",
        ":SPIRVSerializationGen",
        ":SPIRVTargetAndABIStructGen",
        ":SideEffectInterfaces",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "SPIRVPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=SPIRV",
            ],
            "include/mlir/Dialect/SPIRV/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/SPIRV/Transforms/Passes.td",
    deps = [":PassBaseTdFiles"],
)

cc_library(
    name = "SPIRVUtils",
    srcs = glob([
        "lib/Dialect/SPIRV/Utils/*.cpp",
    ]),
    hdrs = glob([
        "include/mlir/Dialect/SPIRV/Utils/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":SPIRVDialect",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "SPIRVConversion",
    srcs = ["lib/Dialect/SPIRV/Transforms/SPIRVConversion.cpp"],
    hdrs = ["include/mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"],
    includes = ["include"],
    deps = [
        ":SPIRVDialect",
        ":Support",
        ":TransformUtils",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "SPIRVTransforms",
    srcs = glob(
        [
            "lib/Dialect/SPIRV/Transforms/*.cpp",
            "lib/Dialect/SPIRV/Transforms/*.h",
        ],
        exclude = ["lib/Dialect/SPIRV/Transforms/SPIRVConversion.cpp"],
    ),
    hdrs = glob(
        ["include/mlir/Dialect/SPIRV/Transforms/*.h"],
        exclude = ["include/mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"],
    ),
    includes = ["include"],
    deps = [
        ":IR",
        ":Pass",
        ":SPIRVConversion",
        ":SPIRVDialect",
        ":SPIRVPassIncGen",
        ":SPIRVUtils",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "StandardToSPIRV",
    srcs = glob([
        "lib/Conversion/StandardToSPIRV/*.cpp",
        "lib/Conversion/StandardToSPIRV/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/StandardToSPIRV/*.h",
    ]),
    includes = [
        "include",
        "lib/Conversion/StandardToSPIRV",
    ],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":MathDialect",
        ":MemRefDialect",
        ":Pass",
        ":SPIRVConversion",
        ":SPIRVDialect",
        ":SPIRVUtils",
        ":StandardOps",
        ":Support",
        ":TensorDialect",
        ":Transforms",
        ":VectorOps",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "SPIRVBinaryUtils",
    srcs = ["lib/Target/SPIRV/SPIRVBinaryUtils.cpp"],
    hdrs = ["include/mlir/Target/SPIRV/SPIRVBinaryUtils.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":SPIRVAttrUtilsGen",
        ":SPIRVDialect",
        ":SPIRVOpsIncGen",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "SPIRVSerialization",
    srcs = [
        "lib/Target/SPIRV/Serialization/Serialization.cpp",
        "lib/Target/SPIRV/Serialization/SerializeOps.cpp",
        "lib/Target/SPIRV/Serialization/Serializer.cpp",
        "lib/Target/SPIRV/Serialization/Serializer.h",
    ],
    hdrs = ["include/mlir/Target/SPIRV/Serialization.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":SPIRVAttrUtilsGen",
        ":SPIRVBinaryUtils",
        ":SPIRVDialect",
        ":SPIRVOpsIncGen",
        ":SPIRVSerializationGen",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "SPIRVDeserialization",
    srcs = glob([
        "lib/Target/SPIRV/Deserialization/*.cpp",
        "lib/Target/SPIRV/Deserialization/*.h",
    ]),
    hdrs = ["include/mlir/Target/SPIRV/Deserialization.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":SPIRVAttrUtilsGen",
        ":SPIRVBinaryUtils",
        ":SPIRVDialect",
        ":SPIRVOpsIncGen",
        ":SPIRVSerializationGen",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "SPIRVModuleCombiner",
    srcs = glob(
        ["lib/Dialect/SPIRV/Linking/ModuleCombiner/*.cpp"],
    ),
    hdrs = ["include/mlir/Dialect/SPIRV/Linking/ModuleCombiner.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":SPIRVDialect",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "SPIRVTranslateRegistration",
    srcs = ["lib/Target/SPIRV/TranslateRegistration.cpp"],
    includes = ["include"],
    deps = [
        ":IR",
        ":Parser",
        ":SPIRVDeserialization",
        ":SPIRVDialect",
        ":SPIRVSerialization",
        ":Support",
        ":Translation",
        "@llvm-project//llvm:Support",
    ],
)

td_library(
    name = "TensorOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Tensor/IR/TensorBase.td",
        "include/mlir/Dialect/Tensor/IR/TensorOps.td",
    ],
    includes = ["include"],
    deps = [
        ":CastInterfacesTdFiles",
        ":ControlFlowInterfacesTdFiles",
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=tensor",
            ],
            "include/mlir/Dialect/Tensor/IR/TensorOpsDialect.h.inc",
        ),
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/Tensor/IR/TensorOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/Tensor/IR/TensorOps.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Tensor/IR/TensorOps.td",
    deps = [":TensorOpsTdFiles"],
)

cc_library(
    name = "TensorDialect",
    srcs = glob(
        [
            "lib/Dialect/Tensor/IR/*.cpp",
            "lib/Dialect/Tensor/IR/*.h",
        ],
    ) + ["include/mlir/Transforms/InliningUtils.h"],
    hdrs = ["include/mlir/Dialect/Tensor/IR/Tensor.h"],
    includes = ["include"],
    deps = [
        ":CastOpInterfaces",
        ":ControlFlowInterfaces",
        ":IR",
        ":SideEffectInterfaces",
        ":Support",
        ":TensorOpsIncGen",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "TensorPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Tensor",
            ],
            "include/mlir/Dialect/Tensor/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Tensor/Transforms/Passes.td",
    deps = [":PassBaseTdFiles"],
)

cc_library(
    name = "TensorTransforms",
    srcs = glob(
        [
            "lib/Dialect/Tensor/Transforms/*.cpp",
            "lib/Dialect/Tensor/Transforms/*.h",
        ],
    ),
    hdrs = ["include/mlir/Dialect/Tensor/Transforms/Passes.h"],
    includes = ["include"],
    deps = [
        ":Async",
        ":EDSC",
        ":IR",
        ":MemRefDialect",
        ":ParallelLoopMapperAttrGen",
        ":Pass",
        ":SCFDialect",
        ":StandardOps",
        ":Support",
        ":TensorDialect",
        ":TensorPassIncGen",
        ":Transforms",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "Rewrite",
    srcs = glob([
        "lib/Rewrite/*.cpp",
        "lib/Rewrite/*.h",
    ]),
    hdrs = glob(["include/mlir/Rewrite/*.h"]),
    includes = ["include"],
    deps = [
        ":Analysis",
        ":IR",
        ":PDLDialect",
        ":PDLInterpDialect",
        ":PDLToPDLInterp",
        ":Pass",
        ":SideEffectInterfaces",
        "@llvm-project//llvm:Support",
    ],
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
        ":Affine",
        ":Analysis",
        ":ControlFlowInterfaces",
        ":IR",
        ":MemRefDialect",
        ":Pass",
        ":Rewrite",
        ":SCFDialect",
        ":SideEffectInterfaces",
        ":StandardOps",
        ":Support",
        ":TransformsPassIncGen",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "DerivedAttributeOpInterfaceIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Interfaces/DerivedAttributeOpInterface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Interfaces/DerivedAttributeOpInterface.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/DerivedAttributeOpInterface.td",
    deps = [":DerivedAttributeOpInterfaceTdFiles"],
)

cc_library(
    name = "DerivedAttributeOpInterface",
    srcs = ["lib/Interfaces/DerivedAttributeOpInterface.cpp"],
    hdrs = ["include/mlir/Interfaces/DerivedAttributeOpInterface.h"],
    includes = ["include"],
    deps = [
        ":DerivedAttributeOpInterfaceIncGen",
        ":IR",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

td_library(
    name = "DataLayoutInterfacesTdFiles",
    srcs = ["include/mlir/Interfaces/DataLayoutInterfaces.td"],
    includes = ["include"],
)

gentbl_cc_library(
    name = "DataLayoutInterfacesIncGen",
    tbl_outs = [
        (
            ["-gen-attr-interface-decls"],
            "include/mlir/Interfaces/DataLayoutAttrInterface.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "include/mlir/Interfaces/DataLayoutAttrInterface.cpp.inc",
        ),
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Interfaces/DataLayoutOpInterface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Interfaces/DataLayoutOpInterface.cpp.inc",
        ),
        (
            ["-gen-type-interface-decls"],
            "include/mlir/Interfaces/DataLayoutTypeInterface.h.inc",
        ),
        (
            ["-gen-type-interface-defs"],
            "include/mlir/Interfaces/DataLayoutTypeInterface.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/DataLayoutInterfaces.td",
    deps = [":OpBaseTdFiles"],
)

gentbl_cc_library(
    name = "LoopLikeInterfaceIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Interfaces/LoopLikeInterface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Interfaces/LoopLikeInterface.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/LoopLikeInterface.td",
    deps = [":LoopLikeInterfaceTdFiles"],
)

gentbl_cc_library(
    name = "VectorInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Interfaces/VectorInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Interfaces/VectorInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/VectorInterfaces.td",
    deps = [":VectorInterfacesTdFiles"],
)

gentbl_cc_library(
    name = "ViewLikeInterfaceIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Interfaces/ViewLikeInterface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Interfaces/ViewLikeInterface.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/ViewLikeInterface.td",
    deps = [":ViewLikeInterfaceTdFiles"],
)

gentbl_cc_library(
    name = "CopyOpInterfaceIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Interfaces/CopyOpInterface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Interfaces/CopyOpInterface.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/CopyOpInterface.td",
    deps = [":CopyOpInterfaceTdFiles"],
)

gentbl_cc_library(
    name = "TransformsPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Transforms",
            ],
            "include/mlir/Transforms/Passes.h.inc",
        ),
        (
            [
                "-gen-pass-capi-header",
                "--prefix=Transforms",
            ],
            "include/mlir/Transforms/Transforms.capi.h.inc",
        ),
        (
            [
                "-gen-pass-capi-impl",
                "--prefix=Transforms",
            ],
            "include/mlir/Transforms/Transforms.capi.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Transforms/Passes.td",
    deps = [":PassBaseTdFiles"],
)

cc_library(
    name = "Transforms",
    srcs = glob([
        "lib/Transforms/*.cpp",
        "lib/Transforms/*.h",
    ]),
    hdrs = glob(["include/mlir/Transforms/*.h"]),
    includes = ["include"],
    deps = [
        ":Affine",
        ":Analysis",
        ":ControlFlowInterfaces",
        ":CopyOpInterface",
        ":IR",
        ":LinalgOps",
        ":LoopLikeInterface",
        ":MemRefDialect",
        ":Pass",
        ":Rewrite",
        ":SCFDialect",
        ":SideEffectInterfaces",
        ":StandardOps",
        ":Support",
        ":TensorDialect",
        ":TransformUtils",
        ":TransformsPassIncGen",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "CommonFolders",
    srcs = [
    ],
    hdrs = ["include/mlir/Dialect/CommonFolders.h"],
    includes = ["include"],
    deps = [
        ":IR",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "SCFToGPU",
    srcs = ["lib/Conversion/SCFToGPU/SCFToGPU.cpp"],
    hdrs = ["include/mlir/Conversion/SCFToGPU/SCFToGPU.h"],
    includes = ["include"],
    deps = [
        ":Affine",
        ":AffineToStandard",
        ":ConversionPassIncGen",
        ":GPUDialect",
        ":GPUTransforms",
        ":IR",
        ":MemRefDialect",
        ":Pass",
        ":SCFDialect",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":Transforms",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "SCFToGPUPass",
    srcs = [
        "lib/Conversion/PassDetail.h",
        "lib/Conversion/SCFToGPU/SCFToGPUPass.cpp",
    ],
    hdrs = ["include/mlir/Conversion/SCFToGPU/SCFToGPUPass.h"],
    includes = ["include"],
    deps = [
        ":Affine",
        ":ComplexDialect",
        ":ConversionPassIncGen",
        ":GPUDialect",
        ":Pass",
        ":SCFDialect",
        ":SCFToGPU",
        ":StandardOps",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "SCFToSPIRV",
    srcs = glob([
        "lib/Conversion/SCFToSPIRV/*.cpp",
        "lib/Conversion/SCFToSPIRV/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/SCFToSPIRV/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":Affine",
        ":ConversionPassIncGen",
        ":IR",
        ":Pass",
        ":SCFDialect",
        ":SPIRVConversion",
        ":SPIRVDialect",
        ":StandardOps",
        ":StandardToSPIRV",
        ":Support",
        ":TransformUtils",
        ":Transforms",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "SCFToOpenMP",
    srcs = [
        "lib/Conversion/PassDetail.h",
        "lib/Conversion/SCFToOpenMP/SCFToOpenMP.cpp",
    ],
    hdrs = ["include/mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"],
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":OpenMPDialect",
        ":Pass",
        ":SCFDialect",
        ":Support",
        ":Transforms",
    ],
)

cc_library(
    name = "SCFToStandard",
    srcs = [
        "lib/Conversion/PassDetail.h",
        "lib/Conversion/SCFToStandard/SCFToStandard.cpp",
    ],
    hdrs = ["include/mlir/Conversion/SCFToStandard/SCFToStandard.h"],
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":LLVMDialect",
        ":Pass",
        ":SCFDialect",
        ":StandardOps",
        ":Support",
        ":TransformUtils",
        ":Transforms",
    ],
)

alias(
    name = "CFGTransforms",
    actual = "SCFToStandard",
)

cc_library(
    name = "StandardToLLVM",
    srcs = [
        "lib/Conversion/PassDetail.h",
        "lib/Conversion/StandardToLLVM/StandardToLLVM.cpp",
    ],
    hdrs = [
        "include/mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h",
        "include/mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h",
    ],
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":DataLayoutInterfaces",
        ":IR",
        ":LLVMDialect",
        ":MathDialect",
        ":MemRefDialect",
        ":Parser",
        ":Pass",
        ":StandardOps",
        ":StandardOpsTransforms",
        ":Support",
        ":TransformUtils",
        ":Transforms",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

alias(
    name = "LLVMTransforms",
    actual = "StandardToLLVM",
)

gentbl_cc_library(
    name = "CallOpInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Interfaces/CallInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Interfaces/CallInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/CallInterfaces.td",
    deps = [":CallInterfacesTdFiles"],
)

cc_library(
    name = "CallOpInterfaces",
    srcs = ["lib/Interfaces/CallInterfaces.cpp"],
    hdrs = ["include/mlir/Interfaces/CallInterfaces.h"],
    includes = ["include"],
    deps = [
        ":CallOpInterfacesIncGen",
        ":IR",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "CastOpInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Interfaces/CastInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Interfaces/CastInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/CastInterfaces.td",
    deps = [":CastInterfacesTdFiles"],
)

cc_library(
    name = "CastOpInterfaces",
    srcs = ["lib/Interfaces/CastInterfaces.cpp"],
    hdrs = ["include/mlir/Interfaces/CastInterfaces.h"],
    includes = ["include"],
    deps = [
        ":CastOpInterfacesIncGen",
        ":IR",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "ControlFlowInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Interfaces/ControlFlowInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Interfaces/ControlFlowInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/ControlFlowInterfaces.td",
    deps = [":ControlFlowInterfacesTdFiles"],
)

cc_library(
    name = "ControlFlowInterfaces",
    srcs = ["lib/Interfaces/ControlFlowInterfaces.cpp"],
    hdrs = ["include/mlir/Interfaces/ControlFlowInterfaces.h"],
    includes = ["include"],
    deps = [
        ":ControlFlowInterfacesIncGen",
        ":IR",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "InferTypeOpInterfaceIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Interfaces/InferTypeOpInterface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Interfaces/InferTypeOpInterface.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/InferTypeOpInterface.td",
    deps = [":InferTypeOpInterfaceTdFiles"],
)

cc_library(
    name = "InferTypeOpInterface",
    srcs = ["lib/Interfaces/InferTypeOpInterface.cpp"],
    hdrs = ["include/mlir/Interfaces/InferTypeOpInterface.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":InferTypeOpInterfaceIncGen",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "SideEffectInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Interfaces/SideEffectInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Interfaces/SideEffectInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Interfaces/SideEffectInterfaces.td",
    deps = [":SideEffectInterfacesTdFiles"],
)

cc_library(
    name = "SideEffectInterfaces",
    srcs = ["lib/Interfaces/SideEffectInterfaces.cpp"],
    hdrs = ["include/mlir/Interfaces/SideEffectInterfaces.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":SideEffectInterfacesIncGen",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

alias(
    name = "SideEffects",
    actual = "SideEffectInterfaces",
)

cc_library(
    name = "Analysis",
    srcs = glob(
        [
            "lib/Analysis/*.cpp",
            "lib/Analysis/*.h",
            "lib/Analysis/*/*.cpp",
            "lib/Analysis/*/*.h",
        ],
        exclude = [
            "lib/Analysis/Vector*.cpp",
            "lib/Analysis/Vector*.h",
        ],
    ),
    hdrs = glob(
        [
            "include/mlir/Analysis/*.h",
            "include/mlir/Analysis/*/*.h",
        ],
        exclude = ["include/mlir/Analysis/Vector*.h"],
    ),
    includes = ["include"],
    deps = [
        ":Affine",
        ":CallOpInterfaces",
        ":ControlFlowInterfaces",
        ":IR",
        ":LinalgOps",
        ":SCFDialect",
        ":SideEffectInterfaces",
        ":StandardOps",
        ":Support",
        ":ViewLikeInterface",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "Translation",
    srcs = glob([
        "lib/Translation/*.cpp",
        "lib/Translation/*.h",
    ]),
    hdrs = ["include/mlir/Translation.h"],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":IR",
        ":Parser",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "ToLLVMIRTranslation",
    srcs = [
        "lib/Target/LLVMIR/DebugTranslation.cpp",
        "lib/Target/LLVMIR/DebugTranslation.h",
        "lib/Target/LLVMIR/ModuleTranslation.cpp",
        "lib/Target/LLVMIR/TypeTranslation.cpp",
    ],
    hdrs = [
        "include/mlir/Target/LLVMIR/Export.h",
        "include/mlir/Target/LLVMIR/LLVMTranslationInterface.h",
        "include/mlir/Target/LLVMIR/ModuleTranslation.h",
        "include/mlir/Target/LLVMIR/TypeTranslation.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMConversionIncGen",
        ":LLVMDialect",
        ":LLVMIRTransforms",
        ":OpenMPDialect",
        ":Support",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:FrontendOpenMP",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
    ],
)

cc_library(
    name = "AMXToLLVMIRTranslation",
    srcs = glob(["lib/Target/LLVMIR/Dialect/AMX/*.cpp"]),
    hdrs = glob(["include/mlir/Target/LLVMIR/Dialect/AMX/*.h"]),
    includes = ["include"],
    deps = [
        ":AMX",
        ":AMXConversionIncGen",
        ":IR",
        ":Support",
        ":ToLLVMIRTranslation",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "X86VectorToLLVMIRTranslation",
    srcs = glob(["lib/Target/LLVMIR/Dialect/X86Vector/*.cpp"]),
    hdrs = glob(["include/mlir/Target/LLVMIR/Dialect/X86Vector/*.h"]),
    includes = ["include"],
    deps = [
        ":IR",
        ":Support",
        ":ToLLVMIRTranslation",
        ":X86Vector",
        ":X86VectorConversionIncGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "ArmNeonToLLVMIRTranslation",
    srcs = glob(["lib/Target/LLVMIR/Dialect/ArmNeon/*.cpp"]),
    hdrs = glob(["include/mlir/Target/LLVMIR/Dialect/ArmNeon/*.h"]),
    includes = ["include"],
    deps = [
        ":ArmNeon",
        ":ArmNeonConversionIncGen",
        ":ArmNeonIncGen",
        ":IR",
        ":Support",
        ":ToLLVMIRTranslation",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "ArmSVEToLLVMIRTranslation",
    srcs = glob(["lib/Target/LLVMIR/Dialect/ArmSVE/*.cpp"]),
    hdrs = glob(["include/mlir/Target/LLVMIR/Dialect/ArmSVE/*.h"]),
    includes = ["include"],
    deps = [
        ":ArmSVE",
        ":ArmSVEConversionIncGen",
        ":IR",
        ":Support",
        ":ToLLVMIRTranslation",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "NVVMToLLVMIRTranslation",
    srcs = glob(["lib/Target/LLVMIR/Dialect/NVVM/*.cpp"]),
    hdrs = glob(["include/mlir/Target/LLVMIR/Dialect/NVVM/*.h"]),
    includes = ["include"],
    deps = [
        ":IR",
        ":NVVMConversionIncGen",
        ":NVVMDialect",
        ":Support",
        ":ToLLVMIRTranslation",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "ROCDLToLLVMIRTranslation",
    srcs = glob(["lib/Target/LLVMIR/Dialect/ROCDL/*.cpp"]),
    hdrs = glob(["include/mlir/Target/LLVMIR/Dialect/ROCDL/*.h"]),
    includes = ["include"],
    deps = [
        ":IR",
        ":ROCDLConversionIncGen",
        ":ROCDLDialect",
        ":Support",
        ":ToLLVMIRTranslation",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "LLVMToLLVMIRTranslation",
    srcs = glob(["lib/Target/LLVMIR/Dialect/LLVMIR/*.cpp"]),
    hdrs = glob(["include/mlir/Target/LLVMIR/Dialect/LLVMIR/*.h"]),
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMConversionIncGen",
        ":LLVMDialect",
        ":Support",
        ":ToLLVMIRTranslation",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "OpenACCToLLVMIRTranslation",
    srcs = glob(["lib/Target/LLVMIR/Dialect/OpenACC/*.cpp"]),
    hdrs = glob(["include/mlir/Target/LLVMIR/Dialect/OpenACC/*.h"]),
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":OpenACCDialect",
        ":OpenACCToLLVM",
        ":Support",
        ":ToLLVMIRTranslation",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:FrontendOpenMP",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "OpenMPToLLVMIRTranslation",
    srcs = glob(["lib/Target/LLVMIR/Dialect/OpenMP/*.cpp"]),
    hdrs = glob(["include/mlir/Target/LLVMIR/Dialect/OpenMP/*.h"]),
    includes = ["include"],
    deps = [
        ":IR",
        ":OpenMPDialect",
        ":Support",
        ":ToLLVMIRTranslation",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:FrontendOpenMP",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "AllToLLVMIRTranslations",
    hdrs = ["include/mlir/Target/LLVMIR/Dialect/All.h"],
    includes = ["include"],
    deps = [
        ":AMXToLLVMIRTranslation",
        ":ArmNeonToLLVMIRTranslation",
        ":ArmSVEToLLVMIRTranslation",
        ":LLVMToLLVMIRTranslation",
        ":NVVMToLLVMIRTranslation",
        ":OpenACCToLLVMIRTranslation",
        ":OpenMPToLLVMIRTranslation",
        ":ROCDLToLLVMIRTranslation",
        ":X86VectorToLLVMIRTranslation",
    ],
)

cc_library(
    name = "ToLLVMIRTranslationRegistration",
    srcs = ["lib/Target/LLVMIR/ConvertToLLVMIR.cpp"],
    includes = ["include"],
    deps = [
        ":AllToLLVMIRTranslations",
        ":IR",
        ":ToLLVMIRTranslation",
        ":Translation",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "FromLLVMIRTranslation",
    srcs = [
        "lib/Target/LLVMIR/ConvertFromLLVMIR.cpp",
    ],
    hdrs = ["include/mlir/Target/LLVMIR/Import.h"],
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMConversionIncGen",
        ":LLVMDialect",
        ":Support",
        ":Translation",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "ExecutionEngine",
    srcs = [
        "include/mlir/ExecutionEngine/CRunnerUtils.h",
        "lib/ExecutionEngine/ExecutionEngine.cpp",
    ],
    hdrs = [
        "include/mlir/ExecutionEngine/ExecutionEngine.h",
        "include/mlir/ExecutionEngine/MemRefUtils.h",
    ],
    includes = ["include"],
    deps = [
        ":AllToLLVMIRTranslations",
        ":IR",
        ":LLVMDialect",
        ":Support",
        ":ToLLVMIRTranslation",
        ":Translation",
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",  # fixdeps: keep
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@llvm-project//llvm:X86Disassembler",  # fixdeps: keep
    ],
)

cc_library(
    name = "ExecutionEngineUtils",
    srcs = ["lib/ExecutionEngine/OptUtils.cpp"],
    hdrs = ["include/mlir/ExecutionEngine/OptUtils.h"],
    includes = ["include"],
    deps = [
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Coroutines",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
    ],
)

# TODO(jpienaar): Update this.
cc_library(
    name = "MlirOptLib",
    srcs = ["lib/Support/MlirOptMain.cpp"],
    hdrs = ["include/mlir/Support/MlirOptMain.h"],
    includes = ["include"],
    deps = [
        ":Analysis",
        ":ConversionPasses",
        ":GPUToGPURuntimeTransforms",
        ":GPUToNVVMTransforms",
        ":GPUToROCDLTransforms",
        ":GPUToSPIRV",
        ":GPUTransforms",
        ":IR",
        ":Parser",
        ":Pass",
        ":SCFTransforms",
        ":ShapeToStandard",
        ":ShapeTransforms",
        ":StandardOpsTransforms",
        ":StandardToLLVM",
        ":StandardToSPIRV",
        ":Support",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "AllTranslations",
    hdrs = ["include/mlir/InitAllTranslations.h"],
    deps = [
        ":FromLLVMIRTranslation",
        ":SPIRVTranslateRegistration",
        ":ToLLVMIRTranslationRegistration",
    ],
)

cc_library(
    name = "MlirTranslateMain",
    srcs = ["tools/mlir-translate/mlir-translate.cpp"],
    deps = [
        ":AllPassesAndDialects",
        ":AllTranslations",
        ":IR",
        ":Parser",
        ":Support",
        ":Translation",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "mlir-translate",
    deps = [":MlirTranslateMain"],
)

cc_library(
    name = "AllPassesAndDialects",
    hdrs = [
        "include/mlir/InitAllDialects.h",
        "include/mlir/InitAllPasses.h",
    ],
    deps = [
        ":AMX",
        ":AMXTransforms",
        ":Affine",
        ":AffinePassIncGen",
        ":AffineToStandard",
        ":AffineTransforms",
        ":ArmNeon",
        ":ArmSVE",
        ":ArmSVETransforms",
        ":Async",
        ":AsyncPassIncGen",
        ":AsyncToLLVM",
        ":AsyncTransforms",
        ":ComplexDialect",
        ":ComplexToLLVM",
        ":ConversionPasses",
        ":DLTIDialect",
        ":GPUDialect",
        ":GPUPassIncGen",
        ":GPUToGPURuntimeTransforms",
        ":GPUToNVVMTransforms",
        ":GPUToROCDLTransforms",
        ":GPUToSPIRV",
        ":GPUToVulkanTransforms",
        ":GPUTransforms",
        ":IR",
        ":LLVMDialect",
        ":LLVMIRTransforms",
        ":LLVMPassIncGen",
        ":LinalgOps",
        ":LinalgPassIncGen",
        ":LinalgToLLVM",
        ":LinalgToSPIRV",
        ":LinalgToStandard",
        ":LinalgTransforms",
        ":MathDialect",
        ":MathToLibm",
        ":MathTransforms",
        ":MemRefDialect",
        ":MemRefTransforms",
        ":NVVMDialect",
        ":OpenACCDialect",
        ":OpenMPDialect",
        ":OpenMPToLLVM",
        ":PDLDialect",
        ":PDLInterpDialect",
        ":PDLToPDLInterp",
        ":QuantOps",
        ":QuantPassIncGen",
        ":ROCDLDialect",
        ":SCFDialect",
        ":SCFPassIncGen",
        ":SCFToGPUPass",
        ":SCFToStandard",
        ":SCFTransforms",
        ":SDBM",
        ":SPIRVDialect",
        ":SPIRVPassIncGen",
        ":SPIRVToLLVM",
        ":SPIRVTransforms",
        ":Shape",
        ":ShapeToStandard",
        ":ShapeTransforms",
        ":ShapeTransformsPassIncGen",
        ":SparseTensor",
        ":SparseTensorTransforms",
        ":StandardOps",
        ":StandardOpsTransforms",
        ":StandardOpsTransformsPassIncGen",
        ":StandardToLLVM",
        ":StandardToSPIRV",
        ":TensorDialect",
        ":TensorTransforms",
        ":TosaDialect",
        ":TosaToLinalg",
        ":Transforms",
        ":TransformsPassIncGen",
        ":VectorOps",
        ":VectorToLLVM",
        ":VectorToROCDL",
        ":VectorToSCF",
        ":VectorToSPIRV",
        ":X86Vector",
        ":X86VectorTransforms",
    ],
)

alias(
    name = "AllPassesAndDialectsNoRegistration",
    actual = ":AllPassesAndDialects",
    deprecation = "Use AllPassesAndDialects instead",
)

cc_binary(
    name = "mlir-lsp-server",
    srcs = ["tools/mlir-lsp-server/mlir-lsp-server.cpp"],
    includes = ["include"],
    deps = [
        ":AllPassesAndDialects",
        ":IR",
        ":MlirLspServerLib",
    ],
)

cc_binary(
    name = "mlir-opt",
    srcs = ["tools/mlir-opt/mlir-opt.cpp"],
    local_defines = ["MLIR_INCLUDE_TESTS"],
    deps = [
        ":AllPassesAndDialects",
        ":Analysis",
        ":IR",
        ":MlirOptLib",
        ":OpenMPDialect",
        ":Pass",
        ":QuantOps",
        ":SCFToGPUPass",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:AllTargetsCodeGens",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir/test:TestAffine",
        "@llvm-project//mlir/test:TestAnalysis",
        "@llvm-project//mlir/test:TestDLTI",
        "@llvm-project//mlir/test:TestDialect",
        "@llvm-project//mlir/test:TestGPU",
        "@llvm-project//mlir/test:TestIR",
        "@llvm-project//mlir/test:TestLinalg",
        "@llvm-project//mlir/test:TestMath",
        "@llvm-project//mlir/test:TestPass",
        "@llvm-project//mlir/test:TestReducer",
        "@llvm-project//mlir/test:TestRewrite",
        "@llvm-project//mlir/test:TestSCF",
        "@llvm-project//mlir/test:TestSPIRV",
        "@llvm-project//mlir/test:TestShapeDialect",
        "@llvm-project//mlir/test:TestStandardOps",
        "@llvm-project//mlir/test:TestStandardToLLVM",
        "@llvm-project//mlir/test:TestTosaDialect",
        "@llvm-project//mlir/test:TestTransforms",
        "@llvm-project//mlir/test:TestTypeDialect",
        "@llvm-project//mlir/test:TestVector",
    ],
)

cc_library(
    name = "MlirJitRunner",
    srcs = ["lib/ExecutionEngine/JitRunner.cpp"],
    hdrs = ["include/mlir/ExecutionEngine/JitRunner.h"],
    includes = ["include"],
    deps = [
        ":AllPassesAndDialects",
        ":ExecutionEngine",
        ":ExecutionEngineUtils",
        ":IR",
        ":LLVMDialect",
        ":LLVMToLLVMIRTranslation",
        ":OpenACCToLLVMIRTranslation",
        ":OpenMPToLLVMIRTranslation",
        ":Parser",
        ":Pass",
        ":SCFToStandard",
        ":Support",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "mlir_c_runner_utils",
    srcs = [
        "lib/ExecutionEngine/CRunnerUtils.cpp",
        "lib/ExecutionEngine/SparseUtils.cpp",
    ],
    hdrs = ["include/mlir/ExecutionEngine/CRunnerUtils.h"],
    includes = ["include"],
)

cc_library(
    name = "mlir_async_runtime_api",
    hdrs = ["include/mlir/ExecutionEngine/AsyncRuntime.h"],
    includes = ["include"],
)

cc_library(
    name = "mlir_async_runtime",
    srcs = ["lib/ExecutionEngine/AsyncRuntime.cpp"],
    copts = ["-Dmlir_async_runtime_EXPORTS"],
    deps = [
        ":mlir_async_runtime_api",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "mlir_runner_utils",
    srcs = ["lib/ExecutionEngine/RunnerUtils.cpp"],
    hdrs = ["include/mlir/ExecutionEngine/RunnerUtils.h"],
    includes = ["include"],
    deps = [":mlir_c_runner_utils"],
)

cc_binary(
    name = "mlir-cpu-runner",
    srcs = ["tools/mlir-cpu-runner/mlir-cpu-runner.cpp"],
    linkopts = ["-ldl"],
    deps = [
        ":AllToLLVMIRTranslations",
        ":ExecutionEngineUtils",
        ":IR",
        ":LLVMDialect",
        ":LLVMToLLVMIRTranslation",
        ":MlirJitRunner",
        ":OpenACCToLLVMIRTranslation",
        ":OpenMPToLLVMIRTranslation",
        ":ToLLVMIRTranslation",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:X86AsmParser",
    ],
)

# This target provides the headers from LLVM's Support target without any of
# the symbols. In particular, it does not contain the static registration code
# which may be executed by at most one shared library loaded by ORCJit. Direct
# dependencies need to avoid requiring symbols from LLVMSupport by adding
# copts = ["-DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1"].
#
# Bazel links the dependencies' object files instead of the archives, which
# means that symbols are linked in even if none are used. The LLVM cmake build
# on the other hand links archives (or shared libraries, depending on
# BUILD_SHARED_LIBS), skipping them if none of the symbols are used.
# See also https://reviews.llvm.org/D95613.
cc_headers_only(
    name = "LLVMSupportHeaders",
    src = "@llvm-project//llvm:Support",
)

cc_library(
    name = "mlir_cuda_runtime",
    srcs = ["lib/ExecutionEngine/CudaRuntimeWrappers.cpp"],
    # Prevent needing EnableABIBreakingChecks symbol from LLVMSupport.
    copts = ["-DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1"],
    deps = [
        ":LLVMSupportHeaders",
        ":mlir_c_runner_utils",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:libcuda",
    ],
)

cc_library(
    name = "VulkanRuntime",
    srcs = ["tools/mlir-vulkan-runner/VulkanRuntime.cpp"],
    hdrs = ["tools/mlir-vulkan-runner/VulkanRuntime.h"],
    deps = [
        ":IR",
        ":Pass",
        ":SPIRVDialect",
        ":SideEffectInterfaces",
        ":StandardOps",
        ":Support",
        "@llvm-project//llvm:Support",
        "@vulkan_headers",
        "@vulkan_sdk//:sdk",
    ],
)

cc_binary(
    name = "tools/libvulkan-runtime-wrappers.so",
    srcs = ["tools/mlir-vulkan-runner/vulkan-runtime-wrappers.cpp"],
    linkshared = True,
    deps = [
        ":VulkanRuntime",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "mlir-vulkan-runner",
    srcs = ["tools/mlir-vulkan-runner/mlir-vulkan-runner.cpp"],
    deps = [
        ":ExecutionEngineUtils",
        ":GPUDialect",
        ":GPUToSPIRV",
        ":GPUToVulkanTransforms",
        ":GPUTransforms",
        ":LLVMDialect",
        ":LLVMToLLVMIRTranslation",
        ":MemRefDialect",
        ":MemRefTransforms",
        ":MlirJitRunner",
        ":Pass",
        ":SPIRVDialect",
        ":SPIRVTransforms",
        ":StandardOps",
        ":StandardToLLVM",
        ":StandardToSPIRV",
        ":ToLLVMIRTranslation",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "mlir-spirv-cpu-runner",
    srcs = ["tools/mlir-spirv-cpu-runner/mlir-spirv-cpu-runner.cpp"],
    deps = [
        ":ExecutionEngineUtils",
        ":GPUDialect",
        ":GPUToSPIRV",
        ":GPUTransforms",
        ":IR",
        ":LLVMDialect",
        ":LLVMToLLVMIRTranslation",
        ":MemRefDialect",
        ":MlirJitRunner",
        ":Pass",
        ":SPIRVConversion",
        ":SPIRVDialect",
        ":SPIRVToLLVM",
        ":SPIRVTransforms",
        ":StandardOps",
        ":StandardToLLVM",
        ":ToLLVMIRTranslation",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "TableGen",
    srcs = glob(["lib/TableGen/*.cpp"]),
    hdrs = glob(["include/mlir/TableGen/*.h"]),
    includes = ["include"],
    deps = [
        ":Support",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
    ],
)

cc_library(
    name = "MlirTableGenMain",
    srcs = ["tools/mlir-tblgen/mlir-tblgen.cpp"],
    includes = ["include"],
    deps = [
        ":Support",
        ":TableGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//llvm:config",
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
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//llvm:config",
    ],
)

cc_binary(
    name = "mlir-linalg-ods-gen",
    srcs = [
        "tools/mlir-linalg-ods-gen/mlir-linalg-ods-gen.cpp",
    ],
    linkopts = [
        "-lm",
        "-lpthread",
    ],
    deps = [
        ":IR",
        ":Support",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//llvm:config",
    ],
)

cc_binary(
    name = "mlir-linalg-ods-yaml-gen",
    srcs = [
        "tools/mlir-linalg-ods-gen/mlir-linalg-ods-yaml-gen.cpp",
    ],
    linkopts = [
        "-lm",
        "-lpthread",
    ],
    deps = [
        ":IR",
        ":Parser",
        ":Support",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//llvm:config",
    ],
)

## OpenACC dialect

# TODO(gcmn): This is sticking td files in a cc_library
gentbl_cc_library(
    name = "AccCommonGen",
    includes = ["/llvm/include"],
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-directive-decl"],
            "include/mlir/Dialect/OpenACC/AccCommon.td",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "@llvm-project//llvm:include/llvm/Frontend/OpenACC/ACC.td",
    deps = ["@llvm-project//llvm:acc_td_files"],
)

td_library(
    name = "OpenAccOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/OpenACC/AccCommon.td",
        "include/mlir/Dialect/OpenACC/OpenACCOps.td",
    ],
    includes = ["include"],
    deps = [":OpBaseTdFiles"],
)

gentbl_cc_library(
    name = "OpenACCOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=acc",
            ],
            "include/mlir/Dialect/OpenACC/OpenACCOpsDialect.h.inc",
        ),
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/OpenACC/OpenACCOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/OpenACC/OpenACCOps.cpp.inc",
        ),
        (
            ["-gen-enum-decls"],
            "include/mlir/Dialect/OpenACC/OpenACCOpsEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/mlir/Dialect/OpenACC/OpenACCOpsEnums.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "g3doc/Dialects/OpenACC/OpenACCOps.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/OpenACC/OpenACCOps.td",
    deps = [":OpenAccOpsTdFiles"],
)

cc_library(
    name = "OpenACCDialect",
    srcs = glob(
        [
            "lib/Dialect/OpenACC/IR/*.cpp",
            "lib/Dialect/OpenACC/IR/*.h",
        ],
    ),
    hdrs = glob([
        "include/mlir/Dialect/OpenACC/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":IR",
        ":OpenACCOpsIncGen",
        ":StandardOps",
        "@llvm-project//llvm:Support",
    ],
)

## OpenMP dialect

# TODO(gcmn): This is sticking td files in a cc_library
gentbl_cc_library(
    name = "OmpCommonTdGen",
    includes = ["/llvm/include"],
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-directive-decl"],
            "include/mlir/Dialect/OpenMP/OmpCommon.td",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "@llvm-project//llvm:include/llvm/Frontend/OpenMP/OMP.td",
    deps = [
        ":OpBaseTdFiles",
        "@llvm-project//llvm:omp_td_files",
    ],
)

td_library(
    name = "OpenMPOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/OpenMP/OmpCommon.td",
        "include/mlir/Dialect/OpenMP/OpenMPOps.td",
    ],
    deps = [
        ":LLVMOpsTdFiles",
        ":OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "OpenMPOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/OpenMP/OpenMPOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/OpenMP/OpenMPOps.cpp.inc",
        ),
        (
            ["-gen-enum-decls"],
            "include/mlir/Dialect/OpenMP/OpenMPOpsEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/mlir/Dialect/OpenMP/OpenMPOpsEnums.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=omp",
            ],
            "include/mlir/Dialect/OpenMP/OpenMPOpsDialect.h.inc",
        ),
        (
            ["-gen-op-doc"],
            "g3doc/Dialects/OpenMP/OpenMPOps.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/OpenMP/OpenMPOps.td",
    deps = [":OpenMPOpsTdFiles"],
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
        ":ControlFlowInterfaces",
        ":IR",
        ":LLVMDialect",
        ":OpenMPOpsIncGen",
        ":SideEffectInterfaces",
        ":StandardOps",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "OpenACCToLLVM",
    srcs = glob([
        "lib/Conversion/OpenACCToLLVM/*.cpp",
        "lib/Conversion/OpenACCToLLVM/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/OpenACCToLLVM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":LLVMDialect",
        ":OpenACCDialect",
        ":Pass",
        ":StandardOps",
        ":StandardToLLVM",
        ":Transforms",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "OpenMPToLLVM",
    srcs = glob([
        "lib/Conversion/OpenMPToLLVM/*.cpp",
        "lib/Conversion/OpenMPToLLVM/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/OpenMPToLLVM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":LLVMDialect",
        ":OpenMPDialect",
        ":Pass",
        ":StandardOps",
        ":StandardToLLVM",
        ":Transforms",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

td_library(
    name = "QuantizationOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Quant/QuantOps.td",
        "include/mlir/Dialect/Quant/QuantOpsBase.td",
    ],
    includes = ["include"],
    deps = [
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "QuantOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/Quant/QuantOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/Quant/QuantOps.cpp.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "include/mlir/Dialect/Quant/QuantOpsDialect.h.inc",
        ),
        (
            ["-gen-op-doc"],
            "g3doc/Dialects/QuantOps/QuantOps.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Quant/QuantOps.td",
    deps = [":QuantizationOpsTdFiles"],
)

gentbl_cc_library(
    name = "QuantPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Quant",
            ],
            "include/mlir/Dialect/Quant/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Quant/Passes.td",
    deps = [":PassBaseTdFiles"],
)

cc_library(
    name = "QuantOps",
    srcs = [
        "lib/Dialect/Quant/IR/QuantOps.cpp",
        "lib/Dialect/Quant/IR/QuantTypes.cpp",
        "lib/Dialect/Quant/IR/TypeDetail.h",
        "lib/Dialect/Quant/IR/TypeParser.cpp",
        "lib/Dialect/Quant/Transforms/ConvertConst.cpp",
        "lib/Dialect/Quant/Transforms/ConvertSimQuant.cpp",
        "lib/Dialect/Quant/Transforms/PassDetail.h",
        "lib/Dialect/Quant/Utils/FakeQuantSupport.cpp",
        "lib/Dialect/Quant/Utils/QuantizeUtils.cpp",
        "lib/Dialect/Quant/Utils/UniformSupport.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/Quant/FakeQuantSupport.h",
        "include/mlir/Dialect/Quant/Passes.h",
        "include/mlir/Dialect/Quant/QuantOps.h",
        "include/mlir/Dialect/Quant/QuantTypes.h",
        "include/mlir/Dialect/Quant/QuantizeUtils.h",
        "include/mlir/Dialect/Quant/UniformSupport.h",
    ],
    includes = ["include"],
    deps = [
        ":IR",
        ":Pass",
        ":QuantOpsIncGen",
        ":QuantPassIncGen",
        ":SideEffectInterfaces",
        ":StandardOps",
        ":TransformUtils",
        "@llvm-project//llvm:Support",
    ],
)

td_library(
    name = "LinalgOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Linalg/IR/LinalgBase.td",
        "include/mlir/Dialect/Linalg/IR/LinalgOps.td",
    ],
    includes = ["include"],
    deps = [
        ":ControlFlowInterfacesTdFiles",
        ":InferTypeOpInterfaceTdFiles",
        ":LoopLikeInterfaceTdFiles",
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
        ":ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "LinalgOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/Linalg/IR/LinalgOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=linalg",
            ],
            "include/mlir/Dialect/Linalg/IR/LinalgOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/IR/LinalgOps.td",
    deps = [":LinalgOpsTdFiles"],
)

genlinalg(
    name = "LinalgNamedStructuredOpsTcIncGen",
    src = "include/mlir/Dialect/Linalg/IR/LinalgNamedStructuredOpsSpec.tc",
    linalg_outs = [
        (
            "-gen-impl -o=$@",
            "include/mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.tcgen.cpp.inc",
        ),
        (
            "-gen-ods-decl -o=$@",
            "include/mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.tcgen.td",
        ),
    ],
    linalggen = ":mlir-linalg-ods-gen",
)

genlinalg(
    name = "LinalgNamedStructuredOpsYamlIncGen",
    src = "include/mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yaml",
    linalg_outs = [
        (
            "-o-impl=$@",
            "include/mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yamlgen.cpp.inc",
        ),
        (
            "-o-ods-decl=$@",
            "include/mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yamlgen.td",
        ),
    ],
    linalggen = ":mlir-linalg-ods-yaml-gen",
)

td_library(
    name = "LinalgStructuredOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Linalg/IR/LinalgInterfaces.td",
        "include/mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.tcgen.td",
        "include/mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yamlgen.td",
        "include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td",
    ],
    includes = ["include"],
    deps = [
        ":CopyOpInterfaceTdFiles",
        ":LinalgOpsTdFiles",
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "LinalgStructuredOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td",
    deps = [":LinalgStructuredOpsTdFiles"],
)

gentbl_cc_library(
    name = "LinalgInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Dialect/Linalg/IR/LinalgInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Dialect/Linalg/IR/LinalgInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/IR/LinalgInterfaces.td",
    deps = [":LinalgStructuredOpsTdFiles"],
)

td_library(
    name = "LinalgDocTdFiles",
    srcs = ["include/mlir/Dialect/Linalg/IR/LinalgDoc.td"],
    includes = ["include"],
    deps = [
        ":LinalgOpsTdFiles",
        ":LinalgStructuredOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "LinalgDocIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-doc"],
            "g3doc/Dialects/Linalg/LinalgOps.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/IR/LinalgDoc.td",
    deps = [":LinalgDocTdFiles"],
)

cc_library(
    name = "LinalgToLLVM",
    srcs = glob([
        "lib/Conversion/LinalgToLLVM/*.cpp",
        "lib/Conversion/LinalgToLLVM/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/LinalgToLLVM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":AffineToStandard",
        ":Analysis",
        ":ConversionPassIncGen",
        ":EDSC",
        ":IR",
        ":LLVMDialect",
        ":LinalgOps",
        ":LinalgTransforms",
        ":Pass",
        ":SCFDialect",
        ":SCFToStandard",
        ":StandardOps",
        ":StandardToLLVM",
        ":Support",
        ":Transforms",
        ":VectorToLLVM",
        ":VectorToSCF",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "LinalgToStandard",
    srcs = glob([
        "lib/Conversion/LinalgToStandard/*.cpp",
        "lib/Conversion/LinalgToStandard/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/LinalgToStandard/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":Affine",
        ":ConversionPassIncGen",
        ":IR",
        ":LinalgOps",
        ":LinalgTransforms",
        ":MemRefDialect",
        ":Pass",
        ":SCFDialect",
        ":StandardOps",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "LinalgToSPIRV",
    srcs = glob([
        "lib/Conversion/LinalgToSPIRV/*.cpp",
        "lib/Conversion/LinalgToSPIRV/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/LinalgToSPIRV/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":DialectUtils",
        ":IR",
        ":LinalgOps",
        ":LinalgTransforms",
        ":Pass",
        ":SPIRVConversion",
        ":SPIRVDialect",
        ":StandardOps",
        ":TransformUtils",
    ],
)

cc_library(
    name = "LinalgOps",
    srcs = [
        "lib/Dialect/Linalg/IR/LinalgOps.cpp",
        "lib/Dialect/Linalg/IR/LinalgTypes.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/Linalg/IR/LinalgOps.h",
        "include/mlir/Dialect/Linalg/IR/LinalgTypes.h",
    ],
    includes = ["include"],
    deps = [
        ":Affine",
        ":CopyOpInterface",
        ":DialectUtils",
        ":IR",
        ":InferTypeOpInterface",
        ":LinalgInterfaces",
        ":LinalgInterfacesIncGen",
        ":LinalgNamedStructuredOpsTcIncGen",
        ":LinalgNamedStructuredOpsYamlIncGen",
        ":LinalgOpsIncGen",
        ":LinalgStructuredOpsIncGen",
        ":MemRefDialect",
        ":Parser",
        ":SideEffectInterfaces",
        ":StandardOps",
        ":Support",
        ":TensorDialect",
        ":ViewLikeInterface",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "LinalgPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Linalg",
            ],
            "include/mlir/Dialect/Linalg/Passes.h.inc",
        ),
        (
            [
                "-gen-pass-capi-header",
                "--prefix=Linalg",
            ],
            "include/mlir/Dialect/Linalg/Passes.capi.h.inc",
        ),
        (
            [
                "-gen-pass-capi-impl",
                "--prefix=Linalg",
            ],
            "include/mlir/Dialect/Linalg/Passes.capi.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Linalg/Passes.td",
    deps = [":PassBaseTdFiles"],
)

cc_library(
    name = "LinalgTransforms",
    srcs = glob([
        "lib/Dialect/Linalg/Transforms/*.cpp",
        "lib/Dialect/Linalg/Transforms/*.h",
    ]) + [
        "lib/Dialect/Linalg/Analysis/DependenceAnalysis.cpp",
        "lib/Dialect/Linalg/Utils/Utils.cpp",
    ],
    hdrs = [
        "include/mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h",
        "include/mlir/Dialect/Linalg/Passes.h",
        "include/mlir/Dialect/Linalg/Transforms/CodegenStrategy.h",
        "include/mlir/Dialect/Linalg/Transforms/Hoisting.h",
        "include/mlir/Dialect/Linalg/Transforms/Transforms.h",
        "include/mlir/Dialect/Linalg/Utils/Utils.h",
    ],
    includes = ["include"],
    deps = [
        ":Affine",
        ":AffineUtils",
        ":Analysis",
        ":ComplexDialect",
        ":DialectUtils",
        ":EDSC",
        ":IR",
        ":LinalgOps",
        ":LinalgPassIncGen",
        ":LinalgStructuredOpsIncGen",
        ":MathDialect",
        ":MemRefDialect",
        ":Pass",
        ":SCFDialect",
        ":SCFTransforms",
        ":StandardOps",
        ":StandardOpsTransforms",
        ":Support",
        ":TensorDialect",
        ":TransformUtils",
        ":VectorOps",
        ":VectorToSCF",
        "@llvm-project//llvm:Support",
    ],
)

td_library(
    name = "VectorOpsTdFiles",
    srcs = ["include/mlir/Dialect/Vector/VectorOps.td"],
    includes = ["include"],
    deps = [
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
        ":VectorInterfacesTdFiles",
        ":ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "VectorOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/Vector/VectorOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/Vector/VectorOps.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=vector",
            ],
            "include/mlir/Dialect/Vector/VectorOpsDialect.h.inc",
        ),
        (
            ["-gen-enum-decls"],
            "include/mlir/Dialect/Vector/VectorOpsEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/mlir/Dialect/Vector/VectorOpsEnums.cpp.inc",
        ),
        (
            ["-gen-op-doc"],
            "g3doc/Dialects/Vector/VectorOps.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Vector/VectorOps.td",
    deps = [":VectorOpsTdFiles"],
)

cc_library(
    name = "VectorToLLVM",
    srcs = glob([
        "lib/Conversion/VectorToLLVM/*.cpp",
        "lib/Conversion/VectorToLLVM/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/VectorToLLVM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":AMX",
        ":AMXTransforms",
        ":ArmNeon",
        ":ArmSVE",
        ":ArmSVETransforms",
        ":ConversionPassIncGen",
        ":DialectUtils",
        ":IR",
        ":LLVMDialect",
        ":MemRefDialect",
        ":Pass",
        ":StandardOps",
        ":StandardToLLVM",
        ":Support",
        ":ToLLVMIRTranslation",
        ":Transforms",
        ":VectorOps",
        ":X86Vector",
        ":X86VectorTransforms",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "VectorToSCF",
    srcs = glob([
        "lib/Conversion/VectorToSCF/*.cpp",
        "lib/Conversion/VectorToSCF/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/VectorToSCF/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":Affine",
        ":AffineUtils",
        ":ConversionPassIncGen",
        ":IR",
        ":LLVMDialect",
        ":MemRefDialect",
        ":Pass",
        ":SCFDialect",
        ":StandardOps",
        ":StandardToLLVM",
        ":Support",
        ":Transforms",
        ":VectorOps",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

td_library(
    name = "TosaDialectTdFiles",
    srcs = glob(["include/mlir/Dialect/Tosa/IR/*.td"]),
    deps = [
        ":LoopLikeInterfaceTdFiles",
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "TosaDialectIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/Tosa/IR/TosaOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/Tosa/IR/TosaOps.cpp.inc",
        ),
        (
            ["-gen-struct-attr-decls"],
            "include/mlir/Dialect/Tosa/IR/TosaStructs.h.inc",
        ),
        (
            ["-gen-struct-attr-defs"],
            "include/mlir/Dialect/Tosa/IR/TosaStructs.cpp.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "include/mlir/Dialect/Tosa/IR/TosaOpsDialect.h.inc",
        ),
        (
            ["-gen-op-doc"],
            "g3doc/Dialects/Tosa/TosaOps.md",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Tosa/IR/TosaOps.td",
    deps = [":TosaDialectTdFiles"],
)

gentbl_cc_library(
    name = "TosaInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir/Dialect/Tosa/IR/TosaInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir/Dialect/Tosa/IR/TosaInterfaces.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Tosa/IR/TosaInterfaces.td",
    deps = [":TosaDialectTdFiles"],
)

gentbl_cc_library(
    name = "TosaPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TosaOpt",
            ],
            "include/mlir/Dialect/Tosa/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Tosa/Transforms/Passes.td",
    deps = [":PassBaseTdFiles"],
)

cc_library(
    name = "TosaDialect",
    srcs = glob([
        "lib/Dialect/Tosa/IR/*.cpp",
        "lib/Dialect/Tosa/IR/*.h",
        "lib/Dialect/Tosa/Utils/*.cpp",
        "lib/Dialect/Tosa/Transforms/*.cpp",
    ]),
    hdrs = glob([
        "include/mlir/Dialect/Tosa/IR/*.h",
        "include/mlir/Dialect/Tosa/Utils/*.h",
        "include/mlir/Dialect/Tosa/Transforms/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":Dialect",
        ":IR",
        ":LoopLikeInterface",
        ":Pass",
        ":QuantOps",
        ":SideEffectInterfaces",
        ":StandardOps",
        ":TosaDialectIncGen",
        ":TosaInterfacesIncGen",
        ":TosaPassIncGen",
        ":TransformUtils",
    ],
)

cc_library(
    name = "TosaToLinalg",
    srcs = glob([
        "lib/Conversion/TosaToLinalg/*.cpp",
        "lib/Conversion/TosaToLinalg/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/TosaToLinalg/*.h",
    ]),
    includes = [
        "include",
        "lib/Conversion/TosaToLinalg",
    ],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":LinalgOps",
        ":MathDialect",
        ":MemRefDialect",
        ":Pass",
        ":StandardOps",
        ":TensorDialect",
        ":TosaDialect",
        ":Transforms",
    ],
)

cc_library(
    name = "TosaToSCF",
    srcs = glob([
        "lib/Conversion/TosaToSCF/*.cpp",
        "lib/Conversion/TosaToSCF/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/TosaToSCF/*.h",
    ]),
    includes = [
        "include",
        "lib/Conversion/TosaToSCF",
    ],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":Pass",
        ":SCFDialect",
        ":TensorDialect",
        ":TosaDialect",
        ":Transforms",
    ],
)

cc_library(
    name = "TosaToStandard",
    srcs = glob([
        "lib/Conversion/TosaToStandard/*.cpp",
        "lib/Conversion/TosaToStandard/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/TosaToStandard/*.h",
    ]),
    includes = [
        "include",
        "lib/Conversion/TosaToStandard",
    ],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":Pass",
        ":StandardOps",
        ":TensorDialect",
        ":TosaDialect",
        ":Transforms",
    ],
)

td_library(
    name = "ComplexOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Complex/IR/ComplexBase.td",
        "include/mlir/Dialect/Complex/IR/ComplexOps.td",
    ],
    includes = ["include"],
    deps = [
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
        ":VectorInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "ComplexBaseIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=complex",
            ],
            "include/mlir/Dialect/Complex/IR/ComplexOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Complex/IR/ComplexBase.td",
    deps = [":ComplexOpsTdFiles"],
)

gentbl_cc_library(
    name = "ComplexOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/Complex/IR/ComplexOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/Complex/IR/ComplexOps.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Complex/IR/ComplexOps.td",
    deps = [":ComplexOpsTdFiles"],
)

cc_library(
    name = "ComplexDialect",
    srcs = glob(
        [
            "lib/Dialect/Complex/IR/*.cpp",
            "lib/Dialect/Complex/IR/*.h",
        ],
    ),
    hdrs = ["include/mlir/Dialect/Complex/IR/Complex.h"],
    includes = ["include"],
    deps = [
        ":ComplexBaseIncGen",
        ":ComplexOpsIncGen",
        ":IR",
        ":SideEffectInterfaces",
        ":Support",
        ":VectorInterfaces",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "ComplexToLLVM",
    srcs = glob([
        "lib/Conversion/ComplexToLLVM/*.cpp",
        "lib/Conversion/ComplexToLLVM/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/ComplexToLLVM/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":ComplexDialect",
        ":ConversionPassIncGen",
        ":IR",
        ":LLVMDialect",
        ":Pass",
        ":StandardToLLVM",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "ComplexToStandard",
    srcs = glob([
        "lib/Conversion/ComplexToStandard/*.cpp",
        "lib/Conversion/ComplexToStandard/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/ComplexToStandard/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":ComplexDialect",
        ":ConversionPassIncGen",
        ":IR",
        ":MathDialect",
        ":Pass",
        ":StandardOps",
        ":Transforms",
    ],
)

exports_files(
    [
        "include/mlir/Bindings/Python/Attributes.td",
        "include/mlir/Interfaces/CallInterfaces.h",
        "include/mlir/Interfaces/CallInterfaces.td",
        "include/mlir/Interfaces/CastInterfaces.h",
        "include/mlir/Interfaces/CastInterfaces.td",
        "include/mlir/Interfaces/ControlFlowInterfaces.h",
        "include/mlir/Interfaces/ControlFlowInterfaces.td",
        "include/mlir/Interfaces/CopyOpInterface.td",
        "include/mlir/Interfaces/DataLayoutInterfaces.td",
        "include/mlir/Interfaces/InferTypeOpInterface.td",
        "include/mlir/Interfaces/LoopLikeInterface.td",
        "include/mlir/Interfaces/SideEffectInterfaceBase.td",
        "include/mlir/Interfaces/SideEffectInterfaces.td",
        "include/mlir/Interfaces/VectorInterfaces.td",
        "include/mlir/Interfaces/ViewLikeInterface.td",
        "include/mlir/Dialect/DLTI/DLTIBase.td",
        "include/mlir/Dialect/LLVMIR/LLVMOpBase.td",
        "include/mlir/Dialect/StandardOps/IR/Ops.td",
        "include/mlir/Dialect/Shape/IR/ShapeOps.td",
        "include/mlir/Dialect/Shape/IR/ShapeBase.td",
        "include/mlir/IR/OpAsmInterface.td",
        "include/mlir/IR/OpBase.td",
        "include/mlir/IR/RegionKindInterface.td",
        "include/mlir/IR/SymbolInterfaces.td",
        "include/mlir/IR/TensorEncoding.td",
        "include/mlir/Transforms/InliningUtils.h",
    ],
    visibility = [":friends"],
)

td_library(
    name = "MathOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/Math/IR/MathBase.td",
        "include/mlir/Dialect/Math/IR/MathOps.td",
    ],
    includes = ["include"],
    deps = [
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
        ":VectorInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "MathBaseIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=math",
            ],
            "include/mlir/Dialect/Math/IR/MathOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Math/IR/MathBase.td",
    deps = [":MathOpsTdFiles"],
)

gentbl_cc_library(
    name = "MathOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/Math/IR/MathOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/Math/IR/MathOps.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/Math/IR/MathOps.td",
    deps = [":MathOpsTdFiles"],
)

cc_library(
    name = "MathDialect",
    srcs = glob(
        [
            "lib/Dialect/Math/IR/*.cpp",
            "lib/Dialect/Math/IR/*.h",
        ],
    ),
    hdrs = [
        "include/mlir/Dialect/Math/EDSC/Intrinsics.h",
        "include/mlir/Dialect/Math/IR/Math.h",
        "include/mlir/Transforms/InliningUtils.h",
    ],
    includes = ["include"],
    deps = [
        ":EDSC",
        ":IR",
        ":MathBaseIncGen",
        ":MathOpsIncGen",
        ":SideEffectInterfaces",
        ":Support",
        ":VectorInterfaces",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "MathTransforms",
    srcs = glob([
        "lib/Dialect/Math/Transforms/*.cpp",
        "lib/Dialect/Math/Transforms/*.h",
    ]),
    hdrs = glob(["include/mlir/Dialect/Math/Transforms/*.h"]),
    includes = ["include"],
    deps = [
        ":IR",
        ":LLVMDialect",
        ":MathDialect",
        ":Pass",
        ":SCFDialect",
        ":StandardOps",
        ":Support",
        ":Transforms",
        ":VectorOps",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "MathToLibm",
    srcs = glob([
        "lib/Conversion/MathToLibm/*.cpp",
        "lib/Conversion/MathToLibm/*.h",
    ]) + ["lib/Conversion/PassDetail.h"],
    hdrs = glob([
        "include/mlir/Conversion/MathToLibm/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":ConversionPassIncGen",
        ":IR",
        ":LLVMDialect",
        ":MathDialect",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
        ":VectorOps",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

td_library(
    name = "MemRefOpsTdFiles",
    srcs = [
        "include/mlir/Dialect/MemRef/IR/MemRefBase.td",
        "include/mlir/Dialect/MemRef/IR/MemRefOps.td",
    ],
    includes = ["include"],
    deps = [
        ":CastInterfacesTdFiles",
        ":CopyOpInterfaceTdFiles",
        ":OpBaseTdFiles",
        ":SideEffectInterfacesTdFiles",
        ":ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "MemRefBaseIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=memref",
            ],
            "include/mlir/Dialect/MemRef/IR/MemRefOpsDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/MemRef/IR/MemRefBase.td",
    deps = [":MemRefOpsTdFiles"],
)

gentbl_cc_library(
    name = "MemRefOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir/Dialect/MemRef/IR/MemRefOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir/Dialect/MemRef/IR/MemRefOps.cpp.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/MemRef/IR/MemRefOps.td",
    deps = [":MemRefOpsTdFiles"],
)

cc_library(
    name = "MemRefDialect",
    srcs = glob(
        [
            "lib/Dialect/MemRef/IR/*.cpp",
            "lib/Dialect/MemRef/IR/*.h",
            "lib/Dialect/MemRef/Utils/*.cpp",
        ],
    ),
    hdrs = [
        "include/mlir/Dialect/MemRef/IR/MemRef.h",
        "include/mlir/Dialect/MemRef/Utils/MemRefUtils.h",
    ],
    includes = ["include"],
    deps = [
        ":CopyOpInterface",
        ":EDSC",
        ":IR",
        ":InferTypeOpInterface",
        ":MemRefBaseIncGen",
        ":MemRefOpsIncGen",
        ":StandardOps",
        ":TensorDialect",
        ":ViewLikeInterface",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "MemRefPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=MemRef",
            ],
            "include/mlir/Dialect/MemRef/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/MemRef/Transforms/Passes.td",
    deps = [":PassBaseTdFiles"],
)

cc_library(
    name = "MemRefTransforms",
    srcs = glob(
        [
            "lib/Dialect/MemRef/Transforms/*.cpp",
            "lib/Dialect/MemRef/Transforms/*.h",
        ],
    ),
    hdrs = glob(["include/mlir/Dialect/MemRef/Transforms/*.h"]),
    includes = ["include"],
    deps = [
        ":Affine",
        ":IR",
        ":MemRefDialect",
        ":MemRefPassIncGen",
        ":Pass",
        ":StandardOps",
        ":Support",
        ":Transforms",
        ":VectorOps",
        "@llvm-project//llvm:Support",
    ],
)

td_library(
    name = "DLTIDialectTdFiles",
    srcs = [
        "include/mlir/Dialect/DLTI/DLTI.td",
        "include/mlir/Dialect/DLTI/DLTIBase.td",
    ],
    includes = ["include"],
    deps = [":OpBaseTdFiles"],
)

gentbl_cc_library(
    name = "DLTIBaseIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=dlti",
            ],
            "include/mlir/Dialect/DLTI/DLTIDialect.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Dialect/DLTI/DLTIBase.td",
    deps = [":OpBaseTdFiles"],
)

cc_library(
    name = "DLTIDialect",
    srcs = glob(["lib/Dialect/DLTI/*.cpp"]),
    hdrs = glob(["include/mlir/Dialect/DLTI/*.h"]),
    includes = ["include"],
    deps = [
        ":DLTIBaseIncGen",
        ":DataLayoutInterfaces",
        ":IR",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "ReducerIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Reducer",
            ],
            "include/mlir/Reducer/Passes.h.inc",
        ),
    ],
    tblgen = ":mlir-tblgen",
    td_file = "include/mlir/Reducer/Passes.td",
    deps = [
        ":PassBaseTdFiles",
        ":ReducerTdFiles",
    ],
)

cc_binary(
    name = "mlir-reduce",
    srcs = glob([
        "include/mlir/Reducer/*.h",
        "include/mlir/Reducer/Passes/*.h",
        "tools/mlir-reduce/*.cpp",
        "tools/mlir-reduce/Passes/*.cpp",
    ]) + ["lib/Reducer/Tester.cpp"],
    includes = ["include"],
    local_defines = ["MLIR_INCLUDE_TESTS"],
    stamp = 0,
    deps = [
        ":AllPassesAndDialects",
        ":IR",
        ":Parser",
        ":Pass",
        ":ReducerIncGen",
        ":Support",
        ":Transforms",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir/test:TestDialect",
    ],
)
