load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes.h.inc": [
        "-gen-pass-decls",
        "-name=XlaCpuTransforms",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    visibility = ["//visibility:private"],
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    srcs = [
        "add_loop_unroll_flags.cc",
        "add_reduction_fast_math_flags.cc",
        "expand_float_ops.cc",
        "lower_to_llvm.cc",
        "lower_xla_shared.cc",
        "peel_workgroup_loop.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":passes_inc_gen",
        ":xla_cpu_rewrite_patterns",
        "//xla/backends/cpu/codegen/emitters/ir:xla_cpu",
        "//xla/codegen/emitters/ir:xla",
        "//xla/hlo/analysis:indexing_analysis",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorDialect",
    ],
)

cc_library(
    name = "xla_cpu_rewrite_patterns",
    srcs = ["xla_cpu_rewrite_patterns.cc"],
    hdrs = ["xla_cpu_rewrite_patterns.h"],
    deps = [
        "//xla/backends/cpu/codegen/emitters/ir:xla_cpu",
        "//xla/codegen/emitters/ir:xla",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:UBDialect",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorUtils",
    ],
)
