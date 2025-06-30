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

cc_library(
    name = "atomic_rmw_utils",
    srcs = ["atomic_rmw_utils.cc"],
    hdrs = ["atomic_rmw_utils.h"],
    deps = [
        "//xla/codegen/emitters/ir:xla",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes.h.inc": [
        "-gen-pass-decls",
        "-name=Transforms",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    visibility = ["//visibility:private"],
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    srcs = [
        "erase_dead_functions.cc",
        "expand_float_ops.cc",
        "flatten_tensors.cc",
        "lower_tensors.cc",
        "lower_to_llvm.cc",
        "lower_xla_math_lib.cc",
        "lower_xla_to_scf.cc",
        "merge_pointers_to_same_slice.cc",
        "propagate_slice_indices.cc",
        "unswitch_loops.cc",
        "vectorize_loads_stores.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":atomic_rmw_utils",
        ":convert_pure_call_ops_pass",
        ":passes_inc_gen",
        ":propagate_alias_scopes",
        ":simplify_affine_pass",
        ":simplify_arith_pass",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/codegen/emitters/ir:xla_cpu",
        "//xla/backends/gpu/codegen/emitters/ir:xla_gpu",
        "//xla/codegen:device_spec",
        "//xla/codegen/emitters:elemental_hlo_to_mlir",
        "//xla/codegen/emitters:implicit_arith_op_builder",
        "//xla/codegen/emitters/ir:xla",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:map_mhlo_to_scalar_op",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AMDGPUUtils",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:GPUToROCDLTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MathTransforms",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:VectorTransforms",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "convert_pure_call_ops_pass",
    srcs = [
        "convert_pure_call_ops.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":passes_inc_gen",
        "//xla/codegen/emitters/ir:xla",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "simplify_affine_pass",
    srcs = [
        "simplify_affine.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":passes_inc_gen",
        "//xla/codegen/emitters/ir:xla",
        "//xla/hlo/analysis:indexing_analysis",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "simplify_arith_pass",
    srcs = [
        "simplify_arith.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":passes_inc_gen",
        "//xla/codegen/emitters/ir:xla",
        "//xla/hlo/analysis:indexing_analysis",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "propagate_alias_scopes",
    srcs = ["propagate_alias_scopes.cc"],
    deps = [
        ":passes_inc_gen",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)
