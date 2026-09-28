load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_binary(
    name = "tensor_ir-tblgen",
    srcs = glob([
        "tools/tensor_ir-tblgen/*.cpp",
        "tools/tensor_ir-tblgen/*.h",
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
    name = "TensorIRTdFiles",
    srcs = glob(["include/tensor_ir/Dialect/*.td"]),
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:GPUOpsTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PtrTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorIRTypesIncGen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "-typedefs-dialect=nv_tensor_ir",
            ],
            "include/tensor_ir/Dialect/TensorTypes.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "-typedefs-dialect=nv_tensor_ir",
            ],
            "include/tensor_ir/Dialect/TensorTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tensor_ir/Dialect/TensorTypes.td",
    deps = [":TensorIRTdFiles"],
)

td_library(
    name = "TensorIROptionsTdFiles",
    srcs = [
        "include/tensor_ir/Options/Enums.td",
        "include/tensor_ir/Options/Options.td",
        "include/tensor_ir/Options/OptionsBase.td",
    ],
    includes = ["include"],
    deps = [
        ":TensorIRTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorIROptionsEnumsIncGen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "include/tensor_ir/Options/OptionsEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/tensor_ir/Options/OptionsEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tensor_ir/Options/Enums.td",
    deps = [":TensorIROptionsTdFiles"],
)

gentbl_cc_library(
    name = "TensorIRCompilerOptionsIncGen",
    tbl_outs = [
        (
            ["-gen-component-options"],
            "include/tensor_ir/Options/CompilerOptions.h.inc",
        ),
    ],
    tblgen = ":tensor_ir-tblgen",
    td_file = "include/tensor_ir/Options/Options.td",
    deps = [":TensorIROptionsTdFiles"],
)

cc_library(
    name = "NVTensorIRCompilerOptions",
    srcs = [
        "lib/Options/BytecodeVersionOptions.cpp",
        "lib/Options/CompilerInvocation.cpp",
        "lib/Options/Options.cpp",
    ],
    hdrs = [
        "include/tensor_ir/Options/BytecodeVersionOptions.h",
        "include/tensor_ir/Options/Options.h",
        "include/tensor_ir/Options/OptionsDetail.h",
        "include/tensor_ir/Options/OptionsEnums.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRDialect",
        ":NVTensorIRSupport",
        ":NVTensorIRUtils",
        ":TensorIRCompilerOptionsIncGen",
        ":TensorIROptionsEnumsIncGen",
        "@cuda_tile//:CudaTileBytecode",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "TensorIRDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=nv_tensor_ir",
            ],
            "include/tensor_ir/Dialect/TensorDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=nv_tensor_ir",
            ],
            "include/tensor_ir/Dialect/TensorDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tensor_ir/Dialect/TensorDialect.td",
    deps = [":TensorIRTdFiles"],
)

gentbl_cc_library(
    name = "TensorIROpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tensor_ir/Dialect/TensorOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tensor_ir/Dialect/TensorOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tensor_ir/Dialect/TensorOps.td",
    deps = [":TensorIRTdFiles"],
)

gentbl_cc_library(
    name = "TensorIREnumsIncGen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "include/tensor_ir/Dialect/TensorEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/tensor_ir/Dialect/TensorEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tensor_ir/Dialect/TensorEnums.td",
    deps = [":TensorIRTdFiles"],
)

gentbl_cc_library(
    name = "TensorIROpInterfacesIncGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/tensor_ir/Dialect/TensorOpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/tensor_ir/Dialect/TensorOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tensor_ir/Dialect/TensorInterfaces.td",
    deps = [":TensorIRTdFiles"],
)

gentbl_cc_library(
    name = "TensorIRAttrInterfacesIncGen",
    tbl_outs = [
        (
            ["-gen-attr-interface-decls"],
            "include/tensor_ir/Dialect/TensorAttrInterfaces.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "include/tensor_ir/Dialect/TensorAttrInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tensor_ir/Dialect/TensorAttrInterfaces.td",
    deps = [":TensorIRTdFiles"],
)

gentbl_cc_library(
    name = "TensorIRAttrsIncGen",
    tbl_outs = [
        (
            [
                "-gen-attrdef-decls",
                "-attrdefs-dialect=nv_tensor_ir",
            ],
            "include/tensor_ir/Dialect/TensorAttrs.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
                "-attrdefs-dialect=nv_tensor_ir",
            ],
            "include/tensor_ir/Dialect/TensorAttrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tensor_ir/Dialect/TensorAttrs.td",
    deps = [":TensorIRTdFiles"],
)

gentbl_cc_library(
    name = "TensorIROpsCanonicalizationIncGen",
    strip_include_prefix = "lib/Dialect",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "lib/Dialect/TensorOpsCanonicalization.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/TensorOpsCanonicalization.td",
    deps = [
        ":TensorIRTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorIRTransformPassesIncGen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=NVTensorIRTransform",
            ],
            "include/tensor_ir/Transform/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tensor_ir/Transform/Passes.td",
    deps = [
        ":TensorIRTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "TensorToCudaTileConversionPassIncGen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorToCudaTileConversion",
            ],
            "include/tensor_ir/Conversion/TensorToCudaTile/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tensor_ir/Conversion/TensorToCudaTile/Passes.td",
    deps = [
        ":TensorIRTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "NVTensorIRSupport",
    srcs = ["lib/Support/TCutegen.cpp"],
    hdrs = [
        "include/tensor_ir/Support/Macros.h",
        "include/tensor_ir/Support/Status.h",
        "include/tensor_ir/Support/TCutegen.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "NVTensorIRDialect",
    srcs = [
        "lib/Dialect/Canonicalization.cpp",
        "lib/Dialect/TensorAttrs.cpp",
        "lib/Dialect/TensorDialect.cpp",
        "lib/Dialect/TensorOps.cpp",
        "lib/Dialect/TensorTypes.cpp",
    ],
    hdrs = [
        "include/tensor_ir/Dialect/TensorIR.h",
        "include/tensor_ir/Dialect/TensorIRAttrs.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRSupport",
        ":TensorIRAttrInterfacesIncGen",
        ":TensorIRAttrsIncGen",
        ":TensorIRDialectIncGen",
        ":TensorIREnumsIncGen",
        ":TensorIROpInterfacesIncGen",
        ":TensorIROpsCanonicalizationIncGen",
        ":TensorIROpsIncGen",
        ":TensorIRTypesIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:PtrDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

cc_library(
    name = "NVTensorIRUtils",
    srcs = [
        "lib/Utils/ComputeCapability.cpp",
        "lib/Utils/Utils.cpp",
    ],
    hdrs = [
        "include/tensor_ir/Utils/ComputeCapability.h",
        "include/tensor_ir/Utils/Utils.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRDialect",
        ":NVTensorIRSupport",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "NVTensorIRAnalysis",
    srcs = [
        "lib/Analysis/KernelArgLayout.cpp",
        "lib/Analysis/TileAnalyzer.cpp",
        "lib/Analysis/TileCandidateGenerator.cpp",
    ],
    hdrs = [
        "include/tensor_ir/Analysis/TileAnalyzer.h",
        "include/tensor_ir/Analysis/TileCandidateGenerator.h",
        "include/tensor_ir/Compiler/CudaTile/KernelArgLayout.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRCompilerOptions",
        ":NVTensorIRConversionOptions",
        ":NVTensorIRDialect",
        ":NVTensorIRKernelArgLayout",
        ":NVTensorIRUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "NVTensorIRTransform",
    srcs = glob(["lib/Transform/*.cpp"]),
    hdrs = ["include/tensor_ir/Transform/Passes.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRAnalysis",
        ":NVTensorIRCompilerOptions",
        ":NVTensorIRDialect",
        ":NVTensorIRSupport",
        ":NVTensorIRUtils",
        ":TensorIRTransformPassesIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:PtrDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "NVTensorIRConversionOptions",
    hdrs = ["include/tensor_ir/Conversion/TensorToCudaTile/Options.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRCompilerOptions",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "NVTensorIRKernelArgLayout",
    hdrs = ["include/tensor_ir/Runtime/CudaTile/KernelArgLayout.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRConversionOptions",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "NVTensorIRRuntimeTypes",
    hdrs = ["include/tensor_ir/Runtime/Types.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRSupport",
    ],
)

cc_library(
    name = "NVTensorIRToCudaTileConversion",
    srcs = glob(["lib/Conversion/TensorToCudaTile/*.cpp"]),
    hdrs = [
        "include/tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h",
        "include/tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTileInternal.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRAnalysis",
        ":NVTensorIRCompilerOptions",
        ":NVTensorIRConversionOptions",
        ":NVTensorIRDialect",
        ":NVTensorIRSupport",
        ":NVTensorIRUtils",
        ":TensorToCudaTileConversionPassIncGen",
        "@cuda_tile//:CudaTileDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:PtrDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "NVTensorIRCudaTilePipelines",
    srcs = ["lib/Compiler/CudaTile/Pipelines.cpp"],
    hdrs = ["include/tensor_ir/Compiler/CudaTile/Pipelines.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRConversionOptions",
        ":NVTensorIRDialect",
        ":NVTensorIRToCudaTileConversion",
        ":NVTensorIRTransform",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "NVTensorIRTileIRAssembly",
    srcs = ["lib/Compiler/CudaTile/TileIRAssembly.cpp"],
    hdrs = ["include/tensor_ir/Compiler/CudaTile/TileIRAssembly.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRSupport",
        ":NVTensorIRUtils",
        "@cuda_tile//:CudaTileBytecode",
        "@llvm-project//llvm:Support",
    ],
)
