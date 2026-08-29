load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_binary(
    name = "cuda-tile-tblgen",
    srcs = glob([
        "tools/cuda-tile-tblgen/*.cpp",
        "tools/cuda-tile-tblgen/*.h",
    ]),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//mlir:MlirTableGenMain",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TableGen",
    ],
)

exports_files(["LICENSE"])

td_library(
    name = "CudaTileTdFiles",
    srcs = glob(["include/cuda_tile/Dialect/CudaTile/IR/*.td"]),
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "CudaTileDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/Dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Dialect.td",
    deps = [":CudaTileTdFiles"],
)

gentbl_cc_library(
    name = "CudaTileInterfacesIncGen",
    tbl_outs = [
        (
            ["-gen-attr-interface-decls"],
            "include/cuda_tile/Dialect/CudaTile/IR/AttrInterfaces.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "include/cuda_tile/Dialect/CudaTile/IR/AttrInterfaces.cpp.inc",
        ),
        (
            ["-gen-type-interface-decls"],
            "include/cuda_tile/Dialect/CudaTile/IR/TypeInterfaces.h.inc",
        ),
        (
            ["-gen-type-interface-defs"],
            "include/cuda_tile/Dialect/CudaTile/IR/TypeInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Interfaces.td",
    deps = [":CudaTileTdFiles"],
)

gentbl_cc_library(
    name = "CudaTileTypesIncGen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "-typedefs-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/Types.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "-typedefs-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/Types.cpp.inc",
        ),
        (
            [
                "-gen-type-constraint-decls",
                "-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/TypeConstraints.h.inc",
        ),
        (
            [
                "-gen-type-constraint-defs",
                "-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/TypeConstraints.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Types.td",
    deps = [":CudaTileTdFiles"],
)

gentbl_cc_library(
    name = "CudaTileAttrDefsIncGen",
    tbl_outs = [
        (
            [
                "-gen-attrdef-decls",
                "-attrdefs-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/AttrDefs.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
                "-attrdefs-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/AttrDefs.cpp.inc",
        ),
        (
            [
                "-gen-enum-decls",
                "-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/Enums.h.inc",
        ),
        (
            [
                "-gen-enum-defs",
                "-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/Enums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/AttrDefs.td",
    deps = [":CudaTileTdFiles"],
)

gentbl_cc_library(
    name = "CudaTileRemarksIncGen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "include/cuda_tile/Dialect/CudaTile/IR/TileIRRemarks.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/cuda_tile/Dialect/CudaTile/IR/TileIRRemarks.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Remarks.td",
    deps = [":CudaTileTdFiles"],
)

gentbl_cc_library(
    name = "CudaTileOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/cuda_tile/Dialect/CudaTile/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/cuda_tile/Dialect/CudaTile/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Ops.td",
    deps = [":CudaTileTdFiles"],
)

gentbl_cc_library(
    name = "CudaTileOpsCanonicalizationIncGen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "lib/Dialect/CudaTile/IR/OpsCanonicalization.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/CudaTile/IR/OpsCanonicalization.td",
    deps = [
        ":CudaTileTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

cc_library(
    name = "CudaTileDialect",
    srcs = glob(
        ["lib/Dialect/CudaTile/IR/*.cpp"],
        exclude = ["lib/Dialect/CudaTile/IR/CudaTileTesting.cpp"],
    ),
    hdrs = glob(["include/cuda_tile/Dialect/CudaTile/IR/*.h"]),
    includes = [
        "include",
        "lib/Dialect/CudaTile/IR",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":CudaTileAttrDefsIncGen",
        ":CudaTileDialectIncGen",
        ":CudaTileInterfacesIncGen",
        ":CudaTileOpsCanonicalizationIncGen",
        ":CudaTileOpsIncGen",
        ":CudaTileTypesIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

td_library(
    name = "CudaTileTransformsTdFiles",
    srcs = ["include/cuda_tile/Dialect/CudaTile/Transforms/Passes.td"],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "CudaTileTransformsIncGen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=CudaTile",
            ],
            "include/cuda_tile/Dialect/CudaTile/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/Transforms/Passes.td",
    deps = [":CudaTileTransformsTdFiles"],
)

cc_library(
    name = "CudaTileTransforms",
    srcs = glob(["lib/Dialect/CudaTile/Transforms/*.cpp"]),
    hdrs = glob(["include/cuda_tile/Dialect/CudaTile/Transforms/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":CudaTileDialect",
        ":CudaTileTransformsIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "CudaTileBytecodeOpsIncGen",
    tbl_outs = [
        (
            ["-gen-cuda-tile-bytecode"],
            "include/cuda_tile/Bytecode/Writer/Bytecode.inc",
        ),
        (
            ["-gen-cuda-tile-bytecode-reader"],
            "include/cuda_tile/Bytecode/Reader/BytecodeReader.inc",
        ),
    ],
    tblgen = ":cuda-tile-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Ops.td",
    deps = [":CudaTileTdFiles"],
)

gentbl_cc_library(
    name = "CudaTileBytecodeOpcodesIncGen",
    tbl_outs = [
        (
            ["-gen-cuda-tile-opcodes"],
            "include/cuda_tile/Bytecode/Common/StaticOpcodes.inc",
        ),
    ],
    tblgen = ":cuda-tile-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/BytecodeOpcodes.td",
    deps = [":CudaTileTdFiles"],
)

gentbl_cc_library(
    name = "CudaTileBytecodeTypeIncGen",
    tbl_outs = [
        (
            ["-gen-cuda-tile-type-bytecode"],
            "include/cuda_tile/Bytecode/Writer/TypeBytecode.inc",
        ),
        (
            ["-gen-cuda-tile-type-bytecode-reader"],
            "include/cuda_tile/Bytecode/Reader/TypeBytecodeReader.inc",
        ),
    ],
    tblgen = ":cuda-tile-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/BytecodeTypeOpcodes.td",
    deps = [":CudaTileTdFiles"],
)

gentbl_cc_library(
    name = "CudaTileBytecodeAttrIncGen",
    tbl_outs = [
        (
            ["-gen-cuda-tile-attr-bytecode"],
            "include/cuda_tile/Bytecode/Writer/AttrBytecode.inc",
        ),
    ],
    tblgen = ":cuda-tile-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/BytecodeAttrOpcodes.td",
    deps = [":CudaTileTdFiles"],
)

cc_library(
    name = "CudaTileBytecode",
    srcs = glob([
        "lib/Bytecode/**/*.cpp",
        "lib/Bytecode/**/*.h",
    ]),
    hdrs = glob([
        "include/cuda_tile/Bytecode/**/*.h",
    ]),
    includes = [
        "include",
        "include/cuda_tile/Bytecode/Common",
        "include/cuda_tile/Bytecode/Reader",
        "include/cuda_tile/Bytecode/Writer",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":CudaTileBytecodeAttrIncGen",
        ":CudaTileBytecodeOpcodesIncGen",
        ":CudaTileBytecodeOpsIncGen",
        ":CudaTileBytecodeTypeIncGen",
        ":CudaTileDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TranslateLib",
    ],
)

cc_library(
    name = "CudaTileOptimizer",
    srcs = ["lib/Dialect/CudaTile/Optimizer/CudaTileOptimizer.cpp"],
    hdrs = ["include/cuda_tile/Dialect/CudaTile/Optimizer/CudaTileOptimizer.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":CudaTileBytecode",
        ":CudaTileDialect",
        ":CudaTileTransforms",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
