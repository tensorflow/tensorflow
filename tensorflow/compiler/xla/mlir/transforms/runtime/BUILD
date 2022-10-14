load("//tensorflow:tensorflow.bzl", "if_google", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = [
        "//tensorflow:internal",
        "@tf_runtime//:friends",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = if_google(
        [
            "//learning/brain/experimental/tfrt/autofusion/...",
            "//platforms/xla/service/cpu/...",
            "//tensorflow/compiler/mlir/tensorflow/...",
            "//tensorflow/compiler/mlir/tfrt/...",
            "//tensorflow/compiler/xla/mlir/transforms/google/...",
            "//tensorflow/compiler/xla/mlir/transforms/runtime/tests/...",
            "//tensorflow/compiler/xla/service/cpu/...",
            "@tf_runtime//...",
        ],
        [
            "//platforms/xla/service/cpu/...",
            "//tensorflow/compiler/mlir/tensorflow/...",
            "//tensorflow/compiler/mlir/tfrt/...",
            "//tensorflow/compiler/xla/mlir/transforms/google/...",
            "//tensorflow/compiler/xla/mlir/transforms/runtime/tests/...",
            "//tensorflow/compiler/xla/service/cpu/...",
        ],
    ),
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
        "convert_asserts.cc",
        "convert_custom_calls.cc",
        "export_functions.cc",
        "ordinal_assignment.cc",
        "rt_to_llvm.cc",
    ],
    hdrs = ["passes.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":custom_call_encoding",
        ":passes_inc_gen",
        "//tensorflow/compiler/xla/mlir/ir/runtime:rt",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:tracing",
        "//tensorflow/compiler/xla/runtime:type_id",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "calling_convention",
    srcs = ["calling_convention.cc"],
    hdrs = ["calling_convention.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/xla/mlir/ir/runtime:rt",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Transforms",
    ],
)

tf_cc_test(
    name = "calling_convention_test",
    srcs = ["calling_convention_test.cc"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":calling_convention",
        "//tensorflow/compiler/xla/mlir/ir/runtime:rt",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "compilation_pipeline_cpu",
    srcs = ["compilation_pipeline_cpu.cc"],
    hdrs = ["compilation_pipeline_cpu.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":compilation_pipeline_options",
        ":compiler",
        ":custom_call_encoding",
        ":passes",
        "//tensorflow/compiler/xla/mlir/transforms/math:passes",
        "//tensorflow/compiler/xla/mlir/transforms/memref:passes",
        "//tensorflow/compiler/xla/runtime:compiler_headers",
        "@llvm-project//mlir:AMXToLLVMIRTranslation",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:ArmNeonToLLVMIRTranslation",
        "@llvm-project//mlir:ArmSVEToLLVMIRTranslation",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:AsyncToLLVM",
        "@llvm-project//mlir:AsyncTransforms",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:LinalgToLLVM",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MathToLibm",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:TensorInferTypeOpInterfaceImpl",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:X86VectorToLLVMIRTranslation",
    ],
    alwayslink = 1,  # has pipeline registration
)

cc_library(
    name = "compilation_pipeline_gpu",
    srcs = ["compilation_pipeline_gpu.cc"],
    hdrs = ["compilation_pipeline_gpu.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":compilation_pipeline_options",
        ":compiler",
        ":passes",
        "//tensorflow/compiler/xla/mlir/ir/runtime/tests:testlib",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "//tensorflow/compiler/xla/runtime:compiler_headers",
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
    ],
    alwayslink = 1,  # has pipeline registration
)

cc_library(
    name = "compilation_pipeline_options",
    hdrs = ["compilation_pipeline_options.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":custom_call_encoding",
        "//tensorflow/compiler/xla/runtime:type_id",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "custom_call_encoding",
    srcs = ["custom_call_encoding.cc"],
    hdrs = ["custom_call_encoding.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/xla/mlir/ir/runtime:rt",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:tracing",
        "//tensorflow/compiler/xla/runtime:type_id",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
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
        ":compiler",
        ":passes",
        ":specialization",
        ":type_converter",
        "//tensorflow/compiler/xla/mlir/ir/runtime:rt",
        "//tensorflow/compiler/xla/runtime:arguments",
        "//tensorflow/compiler/xla/runtime:compiler_headers",
        "//tensorflow/compiler/xla/runtime:constraints",
        "//tensorflow/compiler/xla/runtime:errors",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/runtime:symbolic_shape",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ],
)

cc_library(
    name = "specialization",
    srcs = ["specialization.cc"],
    hdrs = ["specialization.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":type_converter",
        "//tensorflow/compiler/xla/mlir/utils/runtime:constraints",
        "//tensorflow/compiler/xla/runtime:arguments",
        "//tensorflow/compiler/xla/runtime:constraints",
        "//tensorflow/compiler/xla/runtime:errors",
        "//tensorflow/compiler/xla/runtime:symbolic_shape",
        "//tensorflow/compiler/xla/runtime:types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "type_converter",
    srcs = ["type_converter.cc"],
    hdrs = ["type_converter.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/mlir/ir/runtime:rt",
        "//tensorflow/compiler/xla/runtime:errors",
        "//tensorflow/compiler/xla/runtime:types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "type_converter_test",
    srcs = ["type_converter_test.cc"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":type_converter",
        "//tensorflow/compiler/xla/runtime:types",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "compiler",
    hdrs = ["compiler.h"],
    compatible_with = get_compatible_with_cloud(),
    visibility = [":friends"],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)
