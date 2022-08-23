load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = [
        "//tensorflow:internal",
        "@tf_runtime//:friends",
    ],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=RuntimeTransforms",
            ],
            "passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    srcs = [
        "convert_to_entrypoint.cc",
        "rt_to_llvm.cc",
    ],
    hdrs = ["passes.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":custom_call_encoding",
        ":passes_inc_gen",
        "//tensorflow/compiler/xla/mlir/ir/runtime:rt_ops",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:type_id",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "calling_convention",
    srcs = ["calling_convention.cc"],
    hdrs = ["calling_convention.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/xla/mlir/ir/runtime:rt_ops",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "compilation_pipeline",
    srcs = ["compilation_pipeline.cc"],
    hdrs = ["compilation_pipeline.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":custom_call_encoding",
        ":passes",
        "//tensorflow/compiler/xla/runtime:type_id",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,  # has pipeline registration
)

cc_library(
    name = "custom_call_encoding",
    srcs = ["custom_call_encoding.cc"],
    hdrs = ["custom_call_encoding.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:type_id",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "jit_compiler",
    srcs = ["jit_compiler.cc"],
    hdrs = ["jit_compiler.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":calling_convention",
        ":passes",
        ":specialization",
        ":type_converter",
        "//tensorflow/compiler/xla/runtime:arguments",
        "//tensorflow/compiler/xla/runtime:constraints",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/runtime:symbolic_shape",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "specialization",
    srcs = ["specialization.cc"],
    hdrs = ["specialization.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/xla/mlir/transforms/runtime:type_converter",
        "//tensorflow/compiler/xla/mlir/utils/runtime:constraints",
        "//tensorflow/compiler/xla/runtime:arguments",
        "//tensorflow/compiler/xla/runtime:constraints",
        "//tensorflow/compiler/xla/runtime:errors",
        "//tensorflow/compiler/xla/runtime:symbolic_shape",
        "//tensorflow/compiler/xla/runtime:types",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TensorDialect",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "type_converter",
    srcs = ["type_converter.cc"],
    hdrs = ["type_converter.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/xla/mlir/ir/runtime:rt_ops",
        "//tensorflow/compiler/xla/runtime:errors",
        "//tensorflow/compiler/xla/runtime:types",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:IR",
        "@tf_runtime//:support",
    ],
)
