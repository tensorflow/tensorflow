load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_cloud")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "mhlo/IR/hlo_ops.td",
    "lhlo/IR/lhlo_ops.td",
])

# Python extension sources.
exports_files(["bindings/python/MlirHloModule.cc"])

filegroup(
    name = "hlo_ops_td_filegroup",
    srcs = glob(["mhlo/IR/*.td"]),
)

td_library(
    name = "hlo_ops_td_files",
    srcs = glob(["mhlo/IR/*.td"]),
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:CopyOpInterfaceTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:MemRefOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:QuantizationOpsTdFiles",
        "@llvm-project//mlir:ShapeOpsTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
        "@stablehlo//:base_td_files",
        "@stablehlo//:chlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "mhlo_pass_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=AllMhlo",
            ],
            "mhlo/transforms/mhlo_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mhlo/transforms/mhlo_passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

gentbl_cc_library(
    name = "lmhlo_pass_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=AllLmhlo",
            ],
            "lhlo/transforms/lmhlo_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lhlo/transforms/lmhlo_passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

gentbl_cc_library(
    name = "hlo_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "mhlo/IR/hlo_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "mhlo/IR/hlo_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mhlo/IR/hlo_ops.td",
    deps = [":hlo_ops_td_files"],
)

gentbl_cc_library(
    name = "hlo_ops_attrs_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "mhlo/IR/hlo_ops_attrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "mhlo/IR/hlo_ops_attrs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mhlo/IR/hlo_ops.td",
    deps = [":hlo_ops_td_files"],
)

gentbl_cc_library(
    name = "hlo_ops_enums_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "mhlo/IR/hlo_ops_enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "mhlo/IR/hlo_ops_enums.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mhlo/IR/hlo_ops.td",
    deps = [":hlo_ops_td_files"],
)

gentbl_cc_library(
    name = "hlo_ops_typedefs_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "--typedefs-dialect=mhlo",
            ],
            "mhlo/IR/hlo_ops_typedefs.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "--typedefs-dialect=mhlo",
            ],
            "mhlo/IR/hlo_ops_typedefs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mhlo/IR/hlo_ops.td",
    deps = [":hlo_ops_td_files"],
)

gentbl_cc_library(
    name = "hlo_ops_pattern_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "mhlo/IR/",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "mhlo/IR/hlo_patterns.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mhlo/IR/hlo_patterns.td",
    deps = [
        ":hlo_ops_td_files",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "lhlo_ops_structs_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "lhlo/IR/lhlo_ops_structs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "lhlo/IR/lhlo_ops_structs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lhlo/IR/lhlo_ops_structs.td",
    deps = [":lhlo_ops_td_files"],
)

gentbl_cc_library(
    name = "lhlo_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "lhlo/IR/lhlo_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "lhlo/IR/lhlo_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lhlo/IR/lhlo_ops.td",
    deps = [":lhlo_ops_td_files"],
)

gentbl_cc_library(
    name = "lhlo_gpu_ops_enums_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "lhlo_gpu/IR/lhlo_gpu_ops_enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "lhlo_gpu/IR/lhlo_gpu_ops_enums.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lhlo_gpu/IR/lhlo_gpu_ops_enums.td",
    deps = [":lhlo_gpu_ops_td_files"],
)

gentbl_cc_library(
    name = "lhlo_gpu_ops_dialect_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "lhlo_gpu/IR/lhlo_gpu_ops_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "lhlo_gpu/IR/lhlo_gpu_ops_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lhlo_gpu/IR/lhlo_gpu_ops_enums.td",
    deps = [":lhlo_gpu_ops_td_files"],
)

gentbl_cc_library(
    name = "lhlo_gpu_ops_attrdefs_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "lhlo_gpu/IR/lhlo_gpu_ops_attrdefs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "lhlo_gpu/IR/lhlo_gpu_ops_attrdefs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lhlo_gpu/IR/lhlo_gpu_ops_enums.td",
    deps = [":lhlo_gpu_ops_td_files"],
)

gentbl_filegroup(
    name = "hlo_ops_doc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-dialect-doc",
                "-dialect=mhlo",
            ],
            "g3doc/hlo_ops.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mhlo/IR/hlo_ops.td",
    deps = [":hlo_ops_td_files"],
)

gentbl_filegroup(
    name = "lhlo_ops_doc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-dialect-doc",
                "-dialect=lmhlo",
            ],
            "g3doc/lhlo_ops.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lhlo/IR/lhlo_ops.td",
    deps = [":lhlo_ops_td_files"],
)

cc_library(
    name = "hlo_ops_common",
    srcs = ["mhlo/IR/hlo_ops_common.cc"],
    hdrs = ["mhlo/IR/hlo_ops_common.h"],
    strip_include_prefix = ".",
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

td_library(
    name = "lhlo_gpu_ops_td_files",
    srcs = glob(["lhlo_gpu/IR/*.td"]),
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        ":hlo_ops_td_files",
        ":lhlo_ops_td_files",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "lhlo_gpu_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "lhlo_gpu/IR/lhlo_gpu_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "lhlo_gpu/IR/lhlo_gpu_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lhlo_gpu/IR/lhlo_gpu_ops.td",
    deps = [":lhlo_gpu_ops_td_files"],
)

#TODO(aminim): revisit the naming and grouping of these rules post-move.
gentbl_cc_library(
    name = "canonicalize_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "mhlo/IR/mhlo_canonicalize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mhlo/IR/mhlo_canonicalize.td",
    deps = [":hlo_ops_td_files"],
)

td_library(
    name = "deallocation_ops_td_files",
    srcs = glob(["deallocation/IR/*.td"]),
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "deallocation_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "deallocation/IR/deallocation_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "deallocation/IR/deallocation_ops.cc.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "deallocation/IR/deallocation_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "deallocation/IR/deallocation_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "deallocation/IR/deallocation_ops.td",
    deps = [":deallocation_ops_td_files"],
)

cc_library(
    name = "deallocation_passes",
    srcs = [
        "deallocation/transforms/buffer_reuse.cc",
        "deallocation/transforms/convert_deallocation_ops_to_llvm.cc",
        "deallocation/transforms/deallocate.cc",
        "deallocation/transforms/deallocation_to_scf.cc",
        "deallocation/transforms/split_alloc_tensors.cc",
    ],
    hdrs = ["deallocation/transforms/passes.h"],
    strip_include_prefix = ".",
    deps = [
        ":deallocation",
        ":deallocation_passes_inc_gen",
        ":deallocation_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "deallocation_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Deallocation",
            ],
            "deallocation/transforms/passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "deallocation/transforms/passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "deallocation",
    srcs = ["deallocation/IR/deallocation_ops.cc"],
    hdrs = ["deallocation/IR/deallocation_ops.h"],
    strip_include_prefix = ".",
    deps = [
        ":deallocation_ops_inc_gen",
        ":deallocation_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
    ],
)

cc_library(
    name = "deallocation_utils",
    srcs = ["deallocation/utils/util.cc"],
    hdrs = ["deallocation/utils/util.h"],
    strip_include_prefix = ".",
    deps = [
        ":deallocation_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
    ],
)

td_library(
    name = "lhlo_ops_td_files",
    srcs = glob(["lhlo/IR/*.td"]),
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        ":hlo_ops_td_files",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:CopyOpInterfaceTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:MemRefOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:ShapeOpsTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "lhlo_structured_interface_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "lhlo/IR/lhlo_structured_interface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "lhlo/IR/lhlo_structured_interface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lhlo/IR/lhlo_structured_interface.td",
    deps = [":lhlo_ops_td_files"],
)

cc_library(
    name = "lhlo_structured_interface",
    srcs = ["lhlo/IR/lhlo_structured_interface.cc"],
    hdrs = [
        "lhlo/IR/lhlo_structured_interface.h",
        "lhlo/IR/lhlo_structured_interface.h.inc",
    ],
    strip_include_prefix = ".",
    deps = [
        ":lhlo_structured_interface_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "convert_op_folder",
    srcs = ["utils/convert_op_folder.cc"],
    hdrs = ["utils/convert_op_folder.h"],
    strip_include_prefix = ".",
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "mlir_hlo",
    srcs = [
        "mhlo/IR/hlo_ops.cc",
        "mhlo/IR/hlo_ops.cc.inc",
        "mhlo/IR/hlo_ops.h.inc",
        "mhlo/IR/hlo_ops_attrs.cc.inc",
        "mhlo/IR/hlo_ops_attrs.h.inc",
        "mhlo/IR/hlo_ops_enums.cc.inc",
        "mhlo/IR/hlo_ops_enums.h.inc",
        "mhlo/IR/hlo_ops_typedefs.cc.inc",
        "mhlo/IR/hlo_ops_typedefs.h.inc",
        "mhlo/IR/mhlo_bytecode.cc",
        "utils/hlo_utils.cc",
    ],
    hdrs = [
        "mhlo/IR/hlo_ops.h",
        "mhlo/IR/mhlo_bytecode.h",
        "utils/hlo_utils.h",
    ],
    strip_include_prefix = ".",
    deps = [
        ":canonicalize_inc_gen",
        ":convert_op_folder",
        ":hlo_ops_attrs_inc_gen",
        ":hlo_ops_common",
        ":hlo_ops_enums_inc_gen",
        ":hlo_ops_inc_gen",
        ":hlo_ops_pattern_inc_gen",
        ":hlo_ops_typedefs_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:base",
        "@stablehlo//:broadcast_utils",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_assembly_format",
        "@stablehlo//:stablehlo_type_inference",
    ],
)

cc_library(
    name = "lhlo",
    srcs = ["lhlo/IR/lhlo_ops.cc"],
    hdrs = [
        "lhlo/IR/lhlo_ops.h",
        "lhlo/IR/lhlo_ops_structs.h",
        "lhlo/utils/lhlo_utils.h",
    ],
    strip_include_prefix = ".",
    deps = [
        ":hlo_ops_common",
        ":lhlo_ops_inc_gen",
        ":lhlo_ops_structs_inc_gen",
        ":lhlo_structured_interface",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:CopyOpInterface",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:ViewLikeInterface",
        "@stablehlo//:stablehlo_type_inference",
    ],
)

cc_library(
    name = "lhlo_gpu",
    srcs = ["lhlo_gpu/IR/lhlo_gpu_ops.cc"],
    hdrs = ["lhlo_gpu/IR/lhlo_gpu_ops.h"],
    strip_include_prefix = ".",
    deps = [
        ":hlo_ops_common",
        ":lhlo",
        ":lhlo_gpu_ops_attrdefs_inc_gen",
        ":lhlo_gpu_ops_dialect_inc_gen",
        ":lhlo_gpu_ops_enums_inc_gen",
        ":lhlo_gpu_ops_inc_gen",
        ":lhlo_gpu_ops_ops",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "lhlo_gpu_ops_ops",
    srcs = ["lhlo_gpu/IR/lhlo_gpu_ops.cc.inc"],
    hdrs = ["lhlo_gpu/IR/lhlo_gpu_ops.h.inc"],
    strip_include_prefix = ".",
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:CopyOpInterface",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

cc_library(
    name = "hlo_dialect_registration",
    srcs = ["mhlo/IR/init.cc"],
    hdrs = ["mhlo/IR/register.h"],
    strip_include_prefix = ".",
    deps = [
        ":mlir_hlo",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SparseTensorDialect",
        # Backward compatibility with the old way of registering CHLO dialect
        "@stablehlo//:chlo_ops",
    ],
)

cc_library(
    name = "mhlo_passes",
    srcs = [
        "mhlo/transforms/broadcast_propagation/broadcast_propagation.cc",
        "mhlo/transforms/chlo_legalize_to_hlo/chlo_legalize_to_hlo_pass.cc",
        "mhlo/transforms/collapse_elementwise_map/collapse_elementwise_map.cc",
        "mhlo/transforms/constraint_fusion/constraint_fusion_pass.cc",
        "mhlo/transforms/convert_to_signless/convert_to_signless_pass.cc",
        "mhlo/transforms/expand_hlo_tuples/expand_hlo_tuples.cc",
        "mhlo/transforms/expand_ops_simplifier/expand_ops_simplifier.cc",
        "mhlo/transforms/group_reduction_dimensions/group_reduction_dimensions.cc",
        "mhlo/transforms/hlo_legalize_shape_ops_to_standard/hlo_legalize_shape_ops_to_standard.cc",
        "mhlo/transforms/hlo_legalize_to_arithmetic/hlo_legalize_to_arithmetic.cc",
        "mhlo/transforms/hlo_legalize_to_lhlo/hlo_legalize_to_lhlo.cc",
        "mhlo/transforms/hlo_legalize_to_memref/hlo_legalize_to_memref.cc",
        "mhlo/transforms/hlo_legalize_to_stablehlo/hlo_legalize_to_stablehlo_pass.cc",
        "mhlo/transforms/legalize_control_flow/legalize_control_flow.cc",
        "mhlo/transforms/legalize_einsum_to_dot_general/legalize_einsum_to_dot_general.cc",
        "mhlo/transforms/legalize_gather_to_torch_index_select/legalize_gather_to_torch_index_select.cc",
        "mhlo/transforms/legalize_mhlo_to_thlo/legalize_mhlo_to_thlo.cc",
        "mhlo/transforms/legalize_shape_computations/legalize_shape_computations.cc",
        "mhlo/transforms/legalize_sort/legalize_sort.cc",
        "mhlo/transforms/legalize_to_linalg/legalize_to_linalg.cc",
        "mhlo/transforms/legalize_to_standard/generated_legalize_to_standard.inc",
        "mhlo/transforms/legalize_to_standard/legalize_to_standard.cc",
        "mhlo/transforms/legalize_trigonometric_to_approximation/legalize_trigonometric_to_approximation.cc",
        "mhlo/transforms/lower_complex/generated_lower_complex.inc",
        "mhlo/transforms/lower_complex/lower_complex.cc",
        "mhlo/transforms/lower_general_dot/lower_general_dot.cc",
        "mhlo/transforms/materialize_broadcasts/materialize_broadcasts.cc",
        "mhlo/transforms/materialize_broadcasts/materialize_broadcasts_pass.cc",
        "mhlo/transforms/merge_assuming_ops/merge_assuming_ops.cc",
        "mhlo/transforms/mhlo_canonicalize_dot/mhlo_canonicalize_dot.cc",
        "mhlo/transforms/mhlo_canonicalize_gather/mhlo_canonicalize_gather.cc",
        "mhlo/transforms/mhlo_canonicalize_reduction/mhlo_canonicalize_reduction.cc",
        "mhlo/transforms/mhlo_canonicalize_scatter/mhlo_canonicalize_scatter.cc",
        "mhlo/transforms/mhlo_flatten_tuple/mhlo_flatten_tuple.cc",
        "mhlo/transforms/mhlo_passes.h.inc",
        "mhlo/transforms/optimize_mhlo/optimize_mhlo.cc",
        "mhlo/transforms/optimize_mhlo/optimize_mhlo_pass.cc",
        "mhlo/transforms/prepare_for_export/prepare_for_export.cc",
        "mhlo/transforms/rank_specialization/rank_specialization.cc",
        "mhlo/transforms/restrict_max_rank/restrict_max_rank.cc",
        "mhlo/transforms/shape_legalize_to_hlo/shape_legalize_to_hlo.cc",
        "mhlo/transforms/shape_reification/shape_reification_pass.cc",
        "mhlo/transforms/shape_simplification/shape_simplification.cc",
        "mhlo/transforms/sink_constants_to_control_flow/sink_constants_to_control_flow.cc",
        "mhlo/transforms/sparse_chlo_legalize_to_linalg/sparse_chlo_legalize_to_linalg.cc",
        "mhlo/transforms/sparse_rewriting/sparse_rewriting.cc",
        "mhlo/transforms/stablehlo_legalize_to_hlo/stablehlo_legalize_to_hlo_pass.cc",
        "mhlo/transforms/symbolic_shape_optimization/symbolic_shape_optimization.cc",
        "mhlo/transforms/test_infer_shaped_type/test_infer_shaped_type_pass.cc",
        "mhlo/transforms/unfuse_batch_norm/unfuse_batch_norm_pass.cc",
    ],
    hdrs = [
        "mhlo/interfaces/bufferizable_op_interface_impl.h",
        "mhlo/transforms/passes.h",
        "mhlo/transforms/rewriters.h",
        "mhlo/utils/legalize_to_linalg_utils.h",
        "mhlo/utils/mhlo_rng_utils.h",
    ],
    strip_include_prefix = ".",
    deps = [
        ":chlo_legalize_to_hlo",
        ":hlo_legalize_to_stablehlo",
        ":legalize_to_linalg_utils",
        ":legalize_to_standard_inc_gen",
        ":lhlo",
        ":lower_complex_inc_gen",
        ":map_chlo_to_hlo_op",
        ":map_hlo_to_lhlo_op",
        ":map_mhlo_to_scalar_op",
        ":mhlo_pass_inc_gen",
        ":mhlo_rng_utils",
        ":mhlo_scatter_gather_utils",
        ":mlir_hlo",
        ":shape_component_analysis",
        ":stablehlo_legalize_to_hlo",
        ":thlo",
        ":thlo_bufferizable_op_interface",
        ":type_conversion",
        ":unfuse_batch_norm",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:LinalgUtils",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "type_conversion",
    srcs = ["mhlo/utils/type_conversion.cc"],
    hdrs = ["mhlo/utils/type_conversion.h"],
    strip_include_prefix = ".",
    deps = [
        ":mlir_hlo",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "map_lmhlo_to_scalar_op",
    hdrs = ["lhlo/transforms/map_lmhlo_to_scalar_op.h"],
    strip_include_prefix = ".",
    deps = [
        ":lhlo",
        ":map_lhlo_to_hlo_op",
        ":map_mhlo_to_scalar_op",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:SCFDialect",
    ],
)

cc_library(
    name = "map_mhlo_to_scalar_op",
    hdrs = ["mhlo/transforms/map_mhlo_to_scalar_op.h"],
    strip_include_prefix = ".",
    deps = [
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:VectorDialect",
    ],
)

cc_library(
    name = "map_chlo_to_hlo_op",
    hdrs = ["mhlo/transforms/map_chlo_to_hlo_op.h"],
    strip_include_prefix = ".",
    deps = [
        ":mlir_hlo",
        "@llvm-project//mlir:IR",
        "@stablehlo//:chlo_ops",
    ],
)

cc_library(
    name = "map_hlo_to_lhlo_op",
    hdrs = ["lhlo/transforms/map_hlo_to_lhlo_op.h"],
    strip_include_prefix = ".",
    deps = [
        ":lhlo",
        ":mlir_hlo",
    ],
)

cc_library(
    name = "map_lhlo_to_hlo_op",
    hdrs = ["lhlo/transforms/map_lhlo_to_hlo_op.h"],
    strip_include_prefix = ".",
    deps = [
        ":lhlo",
        ":mlir_hlo",
    ],
)

cc_library(
    name = "map_stablehlo_to_hlo_op",
    hdrs = ["mhlo/transforms/map_stablehlo_to_hlo_op.h"],
    strip_include_prefix = ".",
    deps = [
        ":mlir_hlo",
        "@llvm-project//mlir:IR",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "lmhlo_passes",
    srcs = [
        "lhlo/transforms/legalize_to_tensor_op/legalize_to_tensor_op.cc",
        "lhlo/transforms/lhlo_legalize_to_affine/lhlo_legalize_to_affine.cc",
        "lhlo/transforms/lhlo_legalize_to_gpu/lhlo_legalize_to_gpu.cc",
        "lhlo/transforms/lhlo_legalize_to_parallel_loops/lhlo_legalize_to_parallel_loops.cc",
        "lhlo/transforms/lmhlo_passes.h.inc",
    ],
    hdrs = ["lhlo/transforms/passes.h"],
    strip_include_prefix = ".",
    deps = [
        ":lhlo",
        ":lmhlo_pass_inc_gen",
        ":map_lmhlo_to_scalar_op",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

cc_library(
    name = "codegen_utils",
    srcs = ["utils/codegen_utils.cc"],
    hdrs = ["utils/codegen_utils.h"],
    strip_include_prefix = ".",
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "placement_utils",
    hdrs = ["utils/placement_utils.h"],
    strip_include_prefix = ".",
    deps = ["@llvm-project//llvm:Support"],
)

cc_library(
    name = "lhlo_elemental_utils",
    srcs = ["lhlo/transforms/lhlo_elemental_utils.cc"],
    hdrs = ["lhlo/transforms/lhlo_elemental_utils.h"],
    strip_include_prefix = ".",
    deps = [
        ":codegen_utils",
        ":lhlo",
        ":map_lmhlo_to_scalar_op",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "legalize_to_linalg_utils",
    srcs = ["mhlo/utils/legalize_to_linalg_utils.cc"],
    hdrs = ["mhlo/utils/legalize_to_linalg_utils.h"],
    strip_include_prefix = ".",
    deps = [
        ":map_mhlo_to_scalar_op",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgUtils",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:chlo_ops",
    ],
)

cc_library(
    name = "mhlo_rng_utils",
    srcs = ["mhlo/utils/mhlo_rng_utils.cc"],
    hdrs = ["mhlo/utils/mhlo_rng_utils.h"],
    strip_include_prefix = ".",
    deps = [
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "mhlo_scatter_gather_utils",
    srcs = ["mhlo/utils/mhlo_scatter_gather_utils.cc"],
    hdrs = ["mhlo/utils/mhlo_scatter_gather_utils.h"],
    strip_include_prefix = ".",
    deps = [
        ":mlir_hlo",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:TensorDialect",
    ],
)

gentbl_cc_library(
    name = "legalize_to_standard_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "mhlo/transforms/",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "mhlo/transforms/legalize_to_standard/generated_legalize_to_standard.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td",
    deps = [
        ":hlo_ops_td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:MathOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "lower_complex_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "mhlo/transforms/",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "mhlo/transforms/lower_complex/generated_lower_complex.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mhlo/transforms/lower_complex/lower_complex_patterns.td",
    deps = [
        ":hlo_ops_td_files",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "unfuse_batch_norm",
    srcs = ["mhlo/transforms/unfuse_batch_norm/unfuse_batch_norm.cc"],
    hdrs = ["mhlo/transforms/rewriters.h"],
    strip_include_prefix = ".",
    deps = [
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "chlo_legalize_to_hlo",
    srcs = ["mhlo/transforms/chlo_legalize_to_hlo/chlo_legalize_to_hlo.cc"],
    hdrs = ["mhlo/transforms/rewriters.h"],
    strip_include_prefix = ".",
    deps = [
        ":chlo_legalize_to_hlo_inc_gen",
        ":map_chlo_to_hlo_op",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:broadcast_utils",
        "@stablehlo//:chlo_ops",
    ],
)

gentbl_cc_library(
    name = "chlo_legalize_to_hlo_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "mhlo/transforms",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "mhlo/transforms/chlo_legalize_to_hlo/generated_chlo_legalize_to_hlo.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mhlo/transforms/chlo_legalize_to_hlo/chlo_legalize_to_hlo_patterns.td",
    deps = [":hlo_ops_td_files"],
)

cc_library(
    name = "hlo_legalize_to_stablehlo",
    srcs = ["mhlo/transforms/hlo_legalize_to_stablehlo/hlo_legalize_to_stablehlo.cc"],
    hdrs = ["mhlo/transforms/rewriters.h"],
    strip_include_prefix = ".",
    deps = [
        ":map_stablehlo_to_hlo_op",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_ops_inc_gen",
    ],
)

cc_library(
    name = "stablehlo_legalize_to_hlo",
    srcs = ["mhlo/transforms/stablehlo_legalize_to_hlo/stablehlo_legalize_to_hlo.cc"],
    hdrs = ["mhlo/transforms/rewriters.h"],
    strip_include_prefix = ".",
    deps = [
        ":map_stablehlo_to_hlo_op",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_ops_inc_gen",
    ],
)

cc_library(
    name = "all_passes",
    srcs = [
        # These are not exposed as headers in the dependent targets, and
        # shouldn't be. Ideally, this entire target should be removed.
        "deallocation/transforms/passes.h.inc",
        "lhlo/transforms/lmhlo_passes.h.inc",
        "gml_st/transforms/passes.h.inc",
        "thlo/transforms/thlo_passes.h.inc",
        "transforms/passes.h.inc",
    ],
    hdrs = [
        "deallocation/transforms/passes.h",
        "gml_st/transforms/passes.h",
        "lhlo/transforms/passes.h",
        "mhlo/transforms/passes.h",
        "thlo/transforms/passes.h",
        "transforms/passes.h",
    ],
    strip_include_prefix = ".",
    deps = [
        ":chlo_legalize_to_hlo",
        ":deallocation_passes",
        ":deallocation_passes_inc_gen",
        ":gml_st_passes",
        ":gml_st_passes_inc_gen",
        ":lhlo",
        ":lmhlo_pass_inc_gen",
        ":lmhlo_passes",
        ":mhlo_pass_inc_gen",
        ":mhlo_passes",
        ":stablehlo_legalize_to_hlo",
        ":thlo_passes",
        ":thlo_passes_inc_gen",
        ":transforms_passes",
        ":transforms_passes_inc_gen",
        ":userange_analysis",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "transforms_passes",
    srcs = [
        "analysis/test_userange_analysis.cc",
        "mhlo/analysis/test_shape_component_analysis.cc",
        "transforms/alloc_to_arg_pass.cc",
        "transforms/buffer_packing.cc",
        "transforms/bufferize.cc",
        "transforms/bufferize_pass.cc",
        "transforms/collapse_parallel_loops_to_1d_pass.cc",
        "transforms/detensorize_scf_ops.cc",
        "transforms/generic_host_to_llvm.cc",
        "transforms/lower_index_cast_pass.cc",
        "transforms/propagate_static_shapes_to_kernel.cc",
        "transforms/test_hlo_transform_dialect_interpreter.cc",
        "transforms/tile_loops_pass.cc",
        "transforms/unbufferize_pass.cc",
        "transforms/unroll_loops.cc",
    ],
    hdrs = [
        "transforms/passes.h",
        "transforms/passes.h.inc",
        "transforms/rewriters.h",
    ],
    strip_include_prefix = ".",
    deps = [
        ":gml_st",
        ":gml_st_passes",
        ":lhlo",
        ":mhlo_passes",
        ":mlir_hlo",
        ":shape_component_analysis",
        ":thlo",
        ":thlo_bufferizable_op_interface",
        ":transforms_passes_inc_gen",
        ":type_conversion",
        ":userange_analysis",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:CopyOpInterface",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgToLLVM",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MathToLibm",
        "@llvm-project//mlir:MathTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:PDLDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:TransformDialectTransforms",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:VectorTransforms",
        "@stablehlo//:chlo_ops",
    ],
)

cc_library(
    name = "transforms_gpu_passes",
    srcs = [
        "transforms/gpu_kernel_lowering_passes.cc",
        "transforms/gpu_passes.cc",
        "transforms/gpu_passes.h.inc",
    ],
    hdrs = ["transforms/gpu_passes.h"],
    strip_include_prefix = ".",
    deps = [
        ":gml_st_passes",
        ":gpu_transforms_passes_inc_gen",
        ":lhlo",
        ":mhlo_passes",
        ":mlir_hlo",
        ":transforms_passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:GPUToROCDLTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:ShapeToStandard",
        "@llvm-project//mlir:TensorInferTypeOpInterfaceImpl",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:VectorTransforms",
    ],
)

gentbl_cc_library(
    name = "gml_st_test_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=GmlStTest",
            ],
            "gml_st/transforms/test_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "gml_st/transforms/test_passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "gml_st_test_passes",
    srcs = [
        "gml_st/transforms/test_passes.cc",
        "gml_st/transforms/test_passes.h.inc",
    ],
    hdrs = ["gml_st/transforms/test_passes.h"],
    strip_include_prefix = ".",
    deps = [
        ":gml_st_passes",
        ":gml_st_test_passes_inc_gen",
        ":gml_st_transforms",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "transforms_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=LMHLOTransforms",
            ],
            "transforms/passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

gentbl_cc_library(
    name = "gpu_transforms_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=LMHLOGPUTransforms",
            ],
            "transforms/gpu_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/gpu_passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "userange_analysis",
    srcs = ["analysis/userange_analysis.cc"],
    hdrs = ["analysis/userange_analysis.h"],
    strip_include_prefix = ".",
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopLikeInterface",
    ],
)

cc_library(
    name = "shape_component_analysis",
    srcs = ["mhlo/analysis/shape_component_analysis.cc"],
    hdrs = ["mhlo/analysis/shape_component_analysis.h"],
    strip_include_prefix = ".",
    deps = [
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "gml_st_passes",
    srcs = [
        "gml_st/transforms/add_debug_info/add_debug_info.cc",
        "gml_st/transforms/collapse_shape/collapse_shape.cc",
        "gml_st/transforms/compose_extract_insert_slice/compose_extract_insert_slice.cc",
        "gml_st/transforms/copy_removal/copy_removal.cc",
        "gml_st/transforms/cpu_tiling/cpu_tiling_pipeline.cc",
        "gml_st/transforms/cpu_tiling/fusion_outlining.cc",
        "gml_st/transforms/cpu_tiling/fusion_planning_for_cpu.cc",
        "gml_st/transforms/cpu_tiling/pack_matmul.cc",
        "gml_st/transforms/cpu_tiling/transform_conv_for_cpu.cc",
        "gml_st/transforms/cpu_tiling/transform_dot_for_cpu.cc",
        "gml_st/transforms/cpu_tiling/transform_map_for_cpu.cc",
        "gml_st/transforms/cpu_tiling/transform_matmul_for_cpu.cc",
        "gml_st/transforms/cpu_tiling/transform_mmt4d_for_cpu.cc",
        "gml_st/transforms/cpu_tiling/transform_pack_for_cpu.cc",
        "gml_st/transforms/cpu_tiling/transform_reduce_for_cpu.cc",
        "gml_st/transforms/cpu_tiling/transform_reverse_for_cpu.cc",
        "gml_st/transforms/cpu_tiling/transform_scatter_for_cpu.cc",
        "gml_st/transforms/cpu_tiling/transform_transpose_for_cpu.cc",
        "gml_st/transforms/fusion/fusion.cc",
        "gml_st/transforms/passes.cc",
        "gml_st/transforms/passes.h.inc",
        "gml_st/transforms/peeling/peeling.cc",
        "gml_st/transforms/rewrite_from_elements_op/rewrite_from_elements_op.cc",
        "gml_st/transforms/rewrite_scf_forall/rewrite_scf_forall.cc",
        "gml_st/transforms/scalarization/scalarization.cc",
        "gml_st/transforms/tiling/tile_by_one.cc",
        "gml_st/transforms/tiling/tiling.cc",
        "gml_st/transforms/tiling_softmax/tiling_softmax.cc",
        "gml_st/transforms/transforms.h",
        "gml_st/transforms/vectorization/lower_vectors.cc",
        "gml_st/transforms/vectorization/vectorization.cc",
        "gml_st/transforms/vectorization/vectorize_copy.cc",
        "gml_st/transforms/vectorization/vectorize_for_cpu.cc",
        "gml_st/utils/linalg_utils.cc",
    ],
    hdrs = [
        "gml_st/transforms/fusion/fusion.h",
        "gml_st/transforms/passes.h",
        "gml_st/transforms/peeling/peeling.h",
        "gml_st/transforms/scalarization/scalarization.h",
        "gml_st/transforms/tiling/tiling.h",
        "gml_st/transforms/vectorization/vectorization.h",
        "gml_st/utils/linalg_utils.h",
    ],
    strip_include_prefix = ".",
    deps = [
        ":gml_st",
        ":gml_st_passes_inc_gen",
        ":gml_st_transforms",
        ":lhlo",
        ":mlir_hlo",
        ":thlo",
        ":type_conversion",
        "@llvm-project//llvm:BinaryFormat",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:LinalgUtils",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorInferTypeOpInterfaceImpl",
        "@llvm-project//mlir:TensorTilingInterfaceImpl",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:TilingInterface",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorToSCF",
        "@llvm-project//mlir:VectorTransforms",
        "@llvm-project//mlir:X86VectorTransforms",
        "@stablehlo//:chlo_ops",
    ],
)

CAPI_HEADERS = [
    "bindings/c/Attributes.h",
    "bindings/c/Dialects.h",
    "bindings/c/Passes.h",
    "bindings/c/Types.h",
]

CAPI_SOURCES = [
    "bindings/c/Attributes.cc",
    "bindings/c/Dialects.cc",
    "bindings/c/Passes.cc",
    "bindings/c/Types.cc",
]

cc_library(
    name = "CAPI",
    srcs = CAPI_SOURCES,
    hdrs = CAPI_HEADERS,
    strip_include_prefix = ".",
    deps = [
        ":all_passes",
        ":mlir_hlo",
        "@llvm-project//mlir:CAPIIR",
    ],
)

# Header-only target, used when using the C API from a separate shared library.
cc_library(
    name = "CAPIHeaders",
    hdrs = CAPI_HEADERS,
    strip_include_prefix = ".",
    deps = ["@llvm-project//mlir:CAPIIRHeaders"],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "CAPIObjects",
    srcs = CAPI_SOURCES,
    hdrs = CAPI_HEADERS,
    strip_include_prefix = ".",
    deps = [
        ":all_passes",
        ":mlir_hlo",
        "@llvm-project//mlir:CAPIIRObjects",
    ],
    alwayslink = True,
)

build_test(
    name = "mlir-hlo-opt_build_test",
    targets = [
        ":mlir-hlo-opt",
    ],
)

cc_binary(
    name = "mlir-hlo-opt",
    srcs = ["tools/mlir-hlo-opt/mlir-hlo-opt.cc"],
    deps = [
        ":all_passes",
        ":deallocation",
        ":gml_st",
        ":gml_st_passes",
        ":gml_st_test_passes",
        ":hlo_dialect_registration",
        ":lhlo",
        ":lhlo_gpu",
        ":thlo",
        ":transforms_gpu_passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@stablehlo//:register",
    ],
)

# Python library.

td_library(
    name = "MhloOpsPyTdFiles",
    srcs = ["@llvm-project//mlir:include/mlir/Bindings/Python/Attributes.td"],
    deps = [
        ":hlo_ops_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_filegroup(
    name = "MhloOpsPyGen",
    tbl_outs = [
        (
            [
                "-gen-python-op-bindings",
                "-bind-dialect=mhlo",
            ],
            "bindings/python/mlir/dialects/_mhlo_ops_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "bindings/python/mlir/dialects/MhloOps.td",
    deps = [":MhloOpsPyTdFiles"],
)

filegroup(
    name = "MhloOpsPyFiles",
    srcs = [
        "bindings/python/mlir/dialects/mhlo.py",
        ":MhloOpsPyGen",
    ],
)

td_library(
    name = "gml_st_ops_td_files",
    srcs = glob(["gml_st/IR/*.td"]),
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:DestinationStyleOpInterfaceTdFiles",
        "@llvm-project//mlir:DialectUtilsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "gml_st_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "gml_st/IR/gml_st_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "gml_st/IR/gml_st_ops.cc.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "gml_st/IR/gml_st_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "gml_st/IR/gml_st_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "gml_st/IR/gml_st_ops.td",
    deps = [":gml_st_ops_td_files"],
)

cc_library(
    name = "gml_st",
    srcs = ["gml_st/IR/gml_st_ops.cc"],
    hdrs = ["gml_st/IR/gml_st_ops.h"],
    strip_include_prefix = ".",
    deps = [
        ":gml_st_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

gentbl_cc_library(
    name = "gml_st_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=GmlSt",
            ],
            "gml_st/transforms/passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "gml_st/transforms/passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "gml_st_transforms",
    srcs = ["gml_st/transforms/transforms.cc"],
    hdrs = ["gml_st/transforms/transforms.h"],
    strip_include_prefix = ".",
    deps = [
        ":gml_st",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
    ],
)

td_library(
    name = "thlo_ops_td_files",
    srcs = glob(["thlo/IR/*.td"]),
    compatible_with = get_compatible_with_cloud(),
    includes = ["."],
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "thlo_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "thlo/IR/thlo_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "thlo/IR/thlo_ops.cc.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "thlo/IR/thlo_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "thlo/IR/thlo_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "thlo/IR/thlo_ops.td",
    deps = [
        ":thlo_ops_td_files",
        "@llvm-project//mlir:DestinationStyleOpInterfaceTdFiles",
        "@llvm-project//mlir:TilingInterfaceTdFiles",
    ],
)

cc_library(
    name = "thlo",
    srcs = ["thlo/IR/thlo_ops.cc"],
    hdrs = ["thlo/IR/thlo_ops.h"],
    strip_include_prefix = ".",
    deps = [
        ":gml_st",
        ":thlo_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DestinationStyleOpInterface",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgUtils",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:TilingInterface",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

cc_library(
    name = "thlo_bufferizable_op_interface",
    srcs = ["thlo/interfaces/bufferizable_op_interface_impl.cc"],
    hdrs = ["thlo/interfaces/bufferizable_op_interface_impl.h"],
    strip_include_prefix = ".",
    deps = [
        ":thlo",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:DestinationStyleOpInterface",
    ],
)

gentbl_cc_library(
    name = "thlo_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=AllThlo",
            ],
            "thlo/transforms/thlo_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "thlo/transforms/thlo_passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "thlo_passes",
    srcs = [
        "thlo/transforms/legalize_sort/legalize_sort.cc",
        "thlo/transforms/thlo_passes.h.inc",
    ],
    hdrs = [
        "thlo/transforms/passes.h",
    ],
    strip_include_prefix = ".",
    deps = [
        ":thlo",
        ":thlo_passes_inc_gen",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

# A light-weight runtime support library, used by MLIR code that results
# after lowering some ops in the vector and sparse tensor dialects.
cc_binary(
    name = "libmlir_c_runner_utils.so",
    linkshared = True,
    linkstatic = False,
    deps = ["@llvm-project//mlir:mlir_c_runner_utils"],
)

cc_library(
    name = "mlir_interpreter_dialects",
    srcs = glob(
        [
            "tools/mlir_interpreter/dialects/*.cc",
        ],
        exclude = ["tools/mlir_interpreter/dialects/util.cc"],
    ),
    strip_include_prefix = ".",
    deps = [
        ":deallocation",
        ":gml_st",
        ":mlir_hlo",
        ":mlir_interpreter_dialect_utils",
        ":mlir_interpreter_framework",
        ":thlo",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_interpreter_dialect_utils",
    srcs = [
        "tools/mlir_interpreter/dialects/util.cc",
    ],
    hdrs = [
        "tools/mlir_interpreter/dialects/comparators.h",
        "tools/mlir_interpreter/dialects/cwise_math.h",
        "tools/mlir_interpreter/dialects/util.h",
    ],
    strip_include_prefix = ".",
    deps = [
        ":mlir_interpreter_framework",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

cc_library(
    name = "mlir_interpreter_framework",
    srcs = [
        "tools/mlir_interpreter/framework/interpreter.cc",
        "tools/mlir_interpreter/framework/interpreter_value.cc",
        "tools/mlir_interpreter/framework/registration.cc",
        "tools/mlir_interpreter/framework/tensor_or_memref.cc",
    ],
    hdrs = [
        "tools/mlir_interpreter/framework/interpreter.h",
        "tools/mlir_interpreter/framework/interpreter_value.h",
        "tools/mlir_interpreter/framework/interpreter_value_util.h",
        "tools/mlir_interpreter/framework/registration.h",
        "tools/mlir_interpreter/framework/tensor_or_memref.h",
    ],
    strip_include_prefix = ".",
    deps = [
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

build_test(
    name = "mlir-interpreter-runner_build_test",
    targets = [
        ":mlir-interpreter-runner",
    ],
)

cc_binary(
    name = "mlir-interpreter-runner",
    srcs = ["tools/mlir_interpreter/mlir-interpreter-runner.cc"],
    deps = [
        ":deallocation",
        ":gml_st",
        ":hlo_dialect_registration",
        ":lhlo",
        ":lhlo_gpu",
        ":mhlo_passes",
        ":mlir_interpreter_dialects",
        ":mlir_interpreter_framework",
        ":thlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirReduceLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
