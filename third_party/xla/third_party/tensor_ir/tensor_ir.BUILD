load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

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
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
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
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:SideEffectInterfaces",
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
        ":NVTensorIRDialect",
        ":NVTensorIRRuntime",
        ":NVTensorIRUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
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
        ":NVTensorIRDialect",
        ":NVTensorIRSupport",
        ":NVTensorIRUtils",
        ":TensorIRTransformPassesIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "NVTensorIRToCudaTileConversion",
    srcs = glob(["lib/Conversion/TensorToCudaTile/*.cpp"]),
    hdrs = [
        "include/tensor_ir/Conversion/TensorToCudaTile/Options.h",
        "include/tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTile.h",
        "include/tensor_ir/Conversion/TensorToCudaTile/TensorToCudaTileInternal.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRAnalysis",
        ":NVTensorIRDialect",
        ":NVTensorIRSupport",
        ":NVTensorIRUtils",
        ":TensorToCudaTileConversionPassIncGen",
        "@cuda_tile//:CudaTileDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
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
        ":NVTensorIRDialect",
        ":NVTensorIRToCudaTileConversion",
        ":NVTensorIRTransform",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "NVTensorIRRuntime",
    srcs = ["lib/Runtime/CudaTileRuntimeKernel.cpp"],
    hdrs = [
        "include/tensor_ir/Runtime/CudaTile/CudaTileRuntimeKernel.h",
        "include/tensor_ir/Runtime/CudaTile/KernelArgLayout.h",
        "include/tensor_ir/Runtime/CudaTile/KernelLaunchHelpers.h",
        "include/tensor_ir/Runtime/CudaTile/RuntimeOperandAccessor.h",
        "include/tensor_ir/Runtime/IRuntimeKernel.h",
        "include/tensor_ir/Runtime/Types.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRCudaApi",
        ":NVTensorIRSupport",
        "@cuda_tile//:CudaTileBytecode",
        "@cuda_tile//:CudaTileDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "NVTensorIRReference",
    srcs = glob(["lib/Reference/*.cpp"]),
    hdrs = [
        "include/tensor_ir/Reference/reference_graph.h",
        "include/tensor_ir/Reference/reference_node.h",
        "include/tensor_ir/Reference/simplified_tensor.h",
        "include/tensor_ir/Reference/tensor_memory.h",
        "lib/Reference/constant_utils.h",
    ],
    includes = [
        "include",
        "lib/Reference",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRCudaApi",
        ":NVTensorIRDialect",
        ":NVTensorIRRuntime",
        ":NVTensorIRSupport",
        ":NVTensorIRUtils",
        "@llvm-project//llvm:Support",
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
        ":NVTensorIRDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "NVTensorIRCompiler",
    srcs = [
        "lib/Compiler/Compiler.cpp",
        "lib/Compiler/CudaTile/CudaTileCompiler.cpp",
        "lib/Compiler/CudaTile/CudaTileFrontend.cpp",
        "lib/Compiler/CudaTile/TileIRAssembly.cpp",
    ],
    hdrs = [
        "include/tensor_ir/Compiler/CompileOptions.h",
        "include/tensor_ir/Compiler/Compiler.h",
        "include/tensor_ir/Compiler/CudaTile/CudaTileCompiler.h",
        "include/tensor_ir/Compiler/CudaTile/CudaTileFrontend.h",
        "include/tensor_ir/Compiler/CudaTile/TileIRAssembly.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRAnalysis",
        ":NVTensorIRCudaTilePipelines",
        ":NVTensorIRDialect",
        ":NVTensorIRRegistration",
        ":NVTensorIRRuntime",
        ":NVTensorIRSupport",
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
    name = "NVTensorIRCAPI",
    srcs = ["lib/CAPI/TensorIR.cpp"],
    hdrs = ["include/tensor_ir-c/TensorIR.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":NVTensorIRCompiler",
        ":NVTensorIRDialect",
        ":NVTensorIRRegistration",
        ":NVTensorIRRuntime",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)
