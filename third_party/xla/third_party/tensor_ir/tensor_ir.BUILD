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
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
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
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "NVTensorIRCompilerCLOptions",
    srcs = ["lib/Options/OptionsCL.cpp"],
    hdrs = ["include/tensor_ir/Options/OptionsCLDetail.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRCompilerOptions",
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

gentbl_cc_library(
    name = "TensorToLinalgConversionPassIncGen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorToLinalg",
            ],
            "include/tensor_ir/Conversion/TensorToLinalg/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tensor_ir/Conversion/TensorToLinalg/Passes.td",
    deps = [
        ":TensorIRTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "NVTensorIRSupport",
    srcs = [
        "lib/Support/TCutegen.cpp",
        "lib/Support/TIROverrideOptions.cpp",
    ],
    hdrs = [
        "include/tensor_ir/Support/Macros.h",
        "include/tensor_ir/Support/Status.h",
        "include/tensor_ir/Support/TCutegen.h",
        "include/tensor_ir/Support/TIROverrideOptions.h",
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
    name = "NVTensorIRCudaApi",
    srcs = ["lib/Support/CudaApi.cpp"],
    hdrs = ["include/tensor_ir/Support/CudaApi.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRSupport",
        "@llvm-project//llvm:Support",
        "@local_config_cuda//cuda:cuda_headers",
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
        "@llvm-project//mlir:BufferizationDialect",
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
        "lib/Utils/LayoutPermutation.cpp",
        "lib/Utils/Utils.cpp",
    ],
    hdrs = [
        "include/tensor_ir/Utils/ComputeCapability.h",
        "include/tensor_ir/Utils/LayoutPermutation.h",
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
    name = "NVTensorIRPolyExprs",
    srcs = ["lib/Analysis/PolyExprs.cpp"],
    hdrs = ["include/tensor_ir/Analysis/PolyExprs.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "NVTensorIRAnalysis",
    srcs = [
        "lib/Analysis/LayoutPropagation.cpp",
        "lib/Analysis/SymbolicLayout.cpp",
        "lib/Analysis/TileAnalyzer.cpp",
        "lib/Analysis/TileCandidateGenerator.cpp",
    ],
    hdrs = [
        "include/tensor_ir/Analysis/LayoutPropagation.h",
        "include/tensor_ir/Analysis/SymbolicLayout.h",
        "include/tensor_ir/Analysis/TileAnalyzer.h",
        "include/tensor_ir/Analysis/TileCandidateGenerator.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRCompilerOptions",
        ":NVTensorIRDialect",
        ":NVTensorIRPolyExprs",
        ":NVTensorIRSupport",
        ":NVTensorIRUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "NVTensorIRTransform",
    srcs = glob(["lib/Transform/*.cpp"]),
    hdrs = [
        "include/tensor_ir/Transform/HostLoweringUtils.h",
        "include/tensor_ir/Transform/Passes.h",
    ],
    includes = ["include"],
    local_defines = [
        "TENSOR_IR_HOST_TARGET_AArch64=1",
        "TENSOR_IR_HOST_TARGET_X86=1",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRAnalysis",
        ":NVTensorIRCompilerOptions",
        ":NVTensorIRDialect",
        ":NVTensorIRSupport",
        ":NVTensorIRUtils",
        ":TensorIRTransformPassesIncGen",
        "@cuda_tile//:CudaTileDialect",
        "@llvm-project//llvm:AArch64CodeGen",
        "@llvm-project//llvm:AArch64Info",
        "@llvm-project//llvm:AArch64UtilsAndDesc",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:X86CodeGen",
        "@llvm-project//llvm:X86Info",
        "@llvm-project//llvm:X86UtilsAndDesc",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithCommonConversion",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:PtrDialect",
        "@llvm-project//mlir:PtrToLLVM",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:UBDialect",
        "@llvm-project//mlir:UBToLLVM",
    ],
)

genrule(
    name = "NVTensorIRArtifactSchemaGen",
    srcs = ["lib/Artifact/Schema/Artifact.fbs"],
    outs = ["include/tensor_ir/Artifact/Schema/Artifact_generated.h"],
    cmd = "$(location @flatbuffers//:flatc) --cpp --cpp-std c++17 " +
          "--scoped-enums --filename-suffix _generated -o $(@D) $<",
    tools = ["@flatbuffers//:flatc"],
)

cc_library(
    name = "NVTensorIRArtifactSchema",
    hdrs = [":NVTensorIRArtifactSchemaGen"],
    includes = ["include"],
    deps = ["@flatbuffers"],
)

cc_library(
    name = "NVTensorIRRuntimeTypes",
    hdrs = [
        "include/tensor_ir/Runtime/HostRuntimeAbi.h",
        "include/tensor_ir/Runtime/Types.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRSupport",
    ],
)

cc_library(
    name = "NVTensorIRArtifact",
    srcs = [
        "lib/Artifact/KernelMetadata.cpp",
        "lib/Artifact/TensorIRArtifact.cpp",
        "lib/Artifact/TensorIRArtifactCodec.cpp",
    ],
    hdrs = [
        "include/tensor_ir/Artifact/KernelMetadata.h",
        "include/tensor_ir/Artifact/TensorIRArtifact.h",
        "include/tensor_ir/Artifact/TensorIRArtifactCodec.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRArtifactSchema",
        ":NVTensorIRCompilerOptions",
        ":NVTensorIRRuntimeTypes",
        ":NVTensorIRSupport",
        ":NVTensorIRUtils",
        "@cuda_tile//:CudaTileBytecode",
        "@cuda_tile//:CudaTileDialect",
        "@flatbuffers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "NVTensorIRToCudaTileConversion",
    srcs = glob([
        "lib/Conversion/TensorToCudaTile/*.cpp",
        "lib/Conversion/TensorToCudaTile/*.h",
    ]),
    hdrs = [
        "include/tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h",
        "include/tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTileInternal.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRAnalysis",
        ":NVTensorIRCompilerOptions",
        ":NVTensorIRDialect",
        ":NVTensorIRSupport",
        ":NVTensorIRUtils",
        ":TensorToCudaTileConversionPassIncGen",
        "@cuda_tile//:CudaTileBytecode",
        "@cuda_tile//:CudaTileDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:PtrDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "NVTensorIRLinalgScalarBuilder",
    srcs = ["lib/Conversion/TensorToLinalg/ScalarBuilder.cpp"],
    hdrs = ["lib/Conversion/TensorToLinalg/ScalarBuilder.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
    ],
)

cc_library(
    name = "NVTensorIRToLinalgConversion",
    srcs = ["lib/Conversion/TensorToLinalg/TensorToLinalg.cpp"],
    hdrs = ["include/tensor_ir/Conversion/TensorToLinalg/TensorToLinalg.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRDialect",
        ":NVTensorIRLinalgScalarBuilder",
        ":TensorToLinalgConversionPassIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithCommonConversion",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "NVTensorIRLinalgPipelines",
    srcs = ["lib/Compiler/Linalg/Pipelines.cpp"],
    hdrs = ["include/tensor_ir/Compiler/Linalg/Pipelines.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRToLinalgConversion",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ConvertToLLVM",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathToFuncs",
        "@llvm-project//mlir:MathToLibm",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "NVTensorIRCudaTilePipelines",
    srcs = ["lib/Compiler/CudaTile/Pipelines.cpp"],
    hdrs = ["include/tensor_ir/Compiler/CudaTile/Pipelines.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRCompilerOptions",
        ":NVTensorIRDialect",
        ":NVTensorIRSupport",
        ":NVTensorIRToCudaTileConversion",
        ":NVTensorIRTransform",
        "@cuda_tile//:CudaTileDialect",
        "@cuda_tile//:CudaTileTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFToControlFlow",
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

cc_library(
    name = "NVTensorIRRuntime",
    srcs = [
        "lib/Runtime/CudaTileRuntimeKernel.cpp",
        "lib/Runtime/KernelLaunchHelpers.cpp",
        "lib/Runtime/NativeHostModule.cpp",
    ],
    hdrs = [
        "include/tensor_ir/Runtime/CudaTile/CudaTileRuntimeKernel.h",
        "include/tensor_ir/Runtime/CudaTile/RuntimeOperandAccessor.h",
        "include/tensor_ir/Runtime/HostRuntimeAbi.h",
        "include/tensor_ir/Runtime/IRuntimeKernel.h",
        "include/tensor_ir/Runtime/KernelLaunchHelpers.h",
        "lib/Runtime/NativeHostModule.h",
    ],
    includes = [
        "include",
        "lib/Runtime",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRArtifact",
        ":NVTensorIRCompilerOptions",
        ":NVTensorIRCudaApi",
        ":NVTensorIRRuntimeTypes",
        ":NVTensorIRSupport",
        ":NVTensorIRTileIRAssembly",
        "@cuda_tile//:CudaTileBytecode",
        "@cuda_tile//:CudaTileDialect",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:OrcShared",
        "@llvm-project//llvm:OrcTargetProcess",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "NVTensorIRRegistration",
    srcs = ["lib/Registration/Registration.cpp"],
    hdrs = ["include/tensor_ir/Registration/Registration.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRCudaTilePipelines",
        ":NVTensorIRDialect",
        ":NVTensorIRLinalgPipelines",
        ":NVTensorIRToCudaTileConversion",
        ":NVTensorIRToLinalgConversion",
        ":NVTensorIRTransform",
        "@cuda_tile//:CudaTileDialect",
        "@cuda_tile//:CudaTileTransforms",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:ArithValueBoundsOpInterfaceImpl",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:CastInterfaces",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToGPURuntimeTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMIRToLLVMTranslation",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:PtrDialect",
        "@llvm-project//mlir:PtrToLLVM",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:UBDialect",
        "@llvm-project//mlir:UBToLLVM",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:VectorTransforms",
    ],
)

cc_library(
    name = "NVTensorIRCompiler",
    srcs = [
        "lib/Compiler/Compiler.cpp",
        "lib/Compiler/CudaTile/CudaTileCompiler.cpp",
        "lib/Compiler/CudaTile/CudaTileFrontend.cpp",
        "lib/Compiler/CudaTile/KernelMetadataExtraction.cpp",
    ],
    hdrs = [
        "include/tensor_ir/Compiler/Compiler.h",
        "include/tensor_ir/Compiler/CudaTile/CudaTileCompiler.h",
        "include/tensor_ir/Compiler/CudaTile/CudaTileFrontend.h",
        "include/tensor_ir/Compiler/CudaTile/KernelMetadataExtraction.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRAnalysis",
        ":NVTensorIRArtifact",
        ":NVTensorIRCompilerOptions",
        ":NVTensorIRCudaTilePipelines",
        ":NVTensorIRDialect",
        ":NVTensorIRRegistration",
        ":NVTensorIRRuntime",
        ":NVTensorIRSupport",
        ":NVTensorIRTileIRAssembly",
        ":NVTensorIRToCudaTileConversion",
        ":NVTensorIRUtils",
        "@cuda_tile//:CudaTileBytecode",
        "@cuda_tile//:CudaTileDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "NVTensorIRTilingAnalysis",
    srcs = [
        "lib/Analysis/Tiling/Arch.cpp",
        "lib/Analysis/Tiling/Enumerate.cpp",
        "lib/Analysis/Tiling/Evaluate.cpp",
    ],
    hdrs = [
        "include/tensor_ir/Analysis/Tiling/Arch.h",
        "include/tensor_ir/Analysis/Tiling/Enumerate.h",
        "include/tensor_ir/Analysis/Tiling/Evaluate.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRAnalysis",
        ":NVTensorIRDialect",
        ":NVTensorIRSupport",
        ":NVTensorIRUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
