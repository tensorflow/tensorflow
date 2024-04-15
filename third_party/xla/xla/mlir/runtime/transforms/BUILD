load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load(
    "@local_tsl//tsl/platform:build_config_root.bzl",
    "if_llvm_aarch64_available",
    "if_llvm_arm_available",
    "if_llvm_powerpc_available",
    "if_llvm_system_z_available",
    "if_llvm_x86_available",
)
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/mlir/runtime:friends"],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
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
        "add_initializations.cc",
        "convert_asserts.cc",
        "convert_custom_calls.cc",
        "export_functions.cc",
        "move_allocas_to_entry_block.cc",
        "ordinal_assignment.cc",
        "rt_to_llvm.cc",
    ],
    hdrs = ["passes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":custom_call_encoding",
        ":passes_inc_gen",
        "//xla/mlir/runtime/ir:rt",
        "//xla/mlir/runtime/utils:custom_calls",
        "//xla/runtime:custom_call",
        "//xla/runtime:logical_result",
        "//xla/runtime:tracing",
        "//xla/runtime:type_id",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "calling_convention",
    srcs = ["calling_convention.cc"],
    hdrs = ["calling_convention.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/mlir/runtime/ir:rt",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

xla_cc_test(
    name = "calling_convention_test",
    srcs = ["calling_convention_test.cc"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":calling_convention",
        "//xla/mlir/runtime/ir:rt",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:TransformUtils",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "compilation_pipeline_cpu",
    srcs = ["compilation_pipeline_cpu.cc"],
    hdrs = ["compilation_pipeline_cpu.h"],
    compatible_with = get_compatible_with_portable(),
    local_defines = select({
        "//xla/service/cpu:experimental_mlir_gpu_enabled": [
            "EXPERIMENTAL_MLIR_GPU=1",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":compilation_pipeline_options",
        ":compiler",
        ":passes",
        "//xla/mlir/backends/cpu/transforms:passes",
        "//xla/mlir/memref/transforms:passes",
        "//xla/mlir/runtime/ir:rt",
        "//xla/mlir_hlo:transforms_passes",
        "//xla/runtime:compiler",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:AsyncToLLVM",
        "@llvm-project//mlir:AsyncTransforms",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Transforms",
        "@local_tsl//tsl/platform:logging",
    ] + select({
        "//xla/service/cpu:experimental_mlir_gpu_enabled": [
            "@llvm-project//mlir:GPUToGPURuntimeTransforms",
            "@llvm-project//mlir:GPUTransforms",
        ],
        "//conditions:default": [],
    }) + if_llvm_aarch64_available([
        "@llvm-project//mlir:ArmSVEToLLVMIRTranslation",
    ]) + if_llvm_arm_available([
        "@llvm-project//mlir:ArmNeonToLLVMIRTranslation",
    ]) + if_llvm_x86_available([
        "@llvm-project//mlir:AMXToLLVMIRTranslation",
        "@llvm-project//mlir:X86VectorToLLVMIRTranslation",
    ]),
    alwayslink = 1,  # has pipeline registration
)

cc_library(
    name = "compilation_pipeline_options",
    hdrs = ["compilation_pipeline_options.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":custom_call_encoding",
        "//xla/runtime:type_id",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "custom_call_encoding",
    srcs = ["custom_call_encoding.cc"],
    hdrs = ["custom_call_encoding.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/mlir/runtime/ir:rt",
        "//xla/runtime:custom_call",
        "//xla/runtime:logical_result",
        "//xla/runtime:memref_view",
        "//xla/runtime:tracing",
        "//xla/runtime:type_id",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/concurrency:async_value",
    ],
)

cc_library(
    name = "jit_compiler",
    srcs = ["jit_compiler.cc"],
    hdrs = ["jit_compiler.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":calling_convention",
        ":compiler",
        ":passes",
        ":specialization",
        ":type_converter",
        "//xla/mlir/runtime/ir:rt",
        "//xla/runtime:arguments",
        "//xla/runtime:compiler",
        "//xla/runtime:constraints",
        "//xla/runtime:errors",
        "//xla/runtime:executable",
        "//xla/runtime:execution_engine",
        "//xla/runtime:logical_result",
        "//xla/runtime:memory_mapper",
        "//xla/runtime:symbolic_shape",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ] + if_llvm_aarch64_available([
        "@llvm-project//llvm:AArch64AsmParser",
    ]) + if_llvm_powerpc_available([
        "@llvm-project//llvm:PowerPCAsmParser",
    ]) + if_llvm_system_z_available([
        "@llvm-project//llvm:SystemZAsmParser",
    ]) + if_llvm_x86_available([
        "@llvm-project//llvm:X86AsmParser",
    ]),
)

cc_library(
    name = "specialization",
    srcs = ["specialization.cc"],
    hdrs = ["specialization.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":type_converter",
        "//xla:xla_data_proto_cc",
        "//xla/mlir/runtime/utils:constraints",
        "//xla/runtime:arguments",
        "//xla/runtime:constraints",
        "//xla/runtime:symbolic_shape",
        "//xla/runtime:types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "type_converter",
    srcs = ["type_converter.cc"],
    hdrs = ["type_converter.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/mlir/runtime/ir:rt",
        "//xla/runtime:types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "type_converter_test",
    srcs = ["type_converter_test.cc"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":type_converter",
        "//xla:xla_data_proto_cc",
        "//xla/runtime:types",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "compiler",
    hdrs = ["compiler.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)
