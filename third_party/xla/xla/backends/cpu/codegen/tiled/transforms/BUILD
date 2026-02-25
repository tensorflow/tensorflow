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
        "-name=XTileCpuTransforms",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    visibility = ["//visibility:private"],
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "lowering_utils",
    srcs = [
        "lowering_utils.cc",
    ],
    hdrs = ["lowering_utils.h"],
    visibility = ["//visibility:private"],
    deps = [
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:VectorDialect",
    ],
)

cc_library(
    name = "passes",
    srcs = [
        "fuse_elementwise_pass.cc",
        "linalg_elementwise_to_vector_pass.cc",
        "lower_xtile_entry.cc",
        "memref_copy_to_loops.cc",
        "shlo_to_vector.cc",
        "tensor_ops_to_bufferizable.cc",
        "unpack_sub_byte_vector_write_pass.cc",
        "vector_to_scalar_pass.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":lowering_utils",
        ":passes_inc_gen",
        ":vectorized_reduce_emitter",
        "//xla/backends/cpu/codegen/emitters/ir:xla_cpu",
        "//xla/codegen/emitters/ir:xla",
        "//xla/codegen/xtile/ir:xtile",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:FuncUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:LinalgUtils",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:MemRefUtils",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorTransforms",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "vectorized_reduce_emitter",
    srcs = ["vectorized_reduce_emitter.cc"],
    hdrs = ["vectorized_reduce_emitter.h"],
    deps = [
        ":lowering_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:VectorDialect",
    ],
)
