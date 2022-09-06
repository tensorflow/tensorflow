load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td",
    "include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.td",
])

# Python extension sources.
exports_files(["python/MlirHloModule.cc"])

filegroup(
    name = "hlo_ops_td_filegroup",
    srcs = glob(["include/mlir-hlo/Dialect/mhlo/IR/*.td"]),
)

td_library(
    name = "hlo_ops_td_files",
    srcs = glob(["include/mlir-hlo/Dialect/mhlo/IR/*.td"]),
    compatible_with = get_compatible_with_cloud(),
    includes = ["include"],
    deps = [
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:base_td_files",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops_td_files",
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
    ],
)

gentbl_cc_library(
    name = "MhloPassIncGen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=MHLO",
            ],
            "include/mlir-hlo/Dialect/mhlo/transforms/mhlo_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/transforms/mhlo_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "LmhloPassIncGen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=LMHLO",
            ],
            "include/mlir-hlo/Dialect/lhlo/transforms/lmhlo_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/lhlo/transforms/lmhlo_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "hlo_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td",
    deps = [":hlo_ops_td_files"],
)

gentbl_cc_library(
    name = "hlo_ops_attrs_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_attrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_attrs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td",
    deps = [":hlo_ops_td_files"],
)

gentbl_cc_library(
    name = "hlo_ops_enums_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_enums.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td",
    deps = [":hlo_ops_td_files"],
)

gentbl_cc_library(
    name = "hlo_ops_typedefs_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "--typedefs-dialect=mhlo",
            ],
            "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_typedefs.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "--typedefs-dialect=mhlo",
            ],
            "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_typedefs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td",
    deps = [":hlo_ops_td_files"],
)

gentbl_cc_library(
    name = "hlo_ops_pattern_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "lib/Dialect/mhlo/IR/",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "lib/Dialect/mhlo/IR/hlo_patterns.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/mhlo/IR/hlo_patterns.td",
    deps = [
        ":hlo_ops_td_files",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "lhlo_ops_structs_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops_structs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops_structs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops_structs.td",
    deps = [":lhlo_ops_td_files"],
)

gentbl_cc_library(
    name = "lhlo_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.td",
    deps = [":lhlo_ops_td_files"],
)

gentbl_cc_library(
    name = "lhlo_gpu_ops_enums_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops_enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops_enums.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops_enums.td",
    deps = [":lhlo_gpu_ops_td_files"],
)

gentbl_cc_library(
    name = "lhlo_gpu_ops_dialect_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops_enums.td",
    deps = [":lhlo_gpu_ops_td_files"],
)

gentbl_cc_library(
    name = "lhlo_gpu_ops_attrdefs_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops_attrdefs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops_attrdefs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops_enums.td",
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
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td",
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
    td_file = "include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.td",
    deps = [":lhlo_ops_td_files"],
)

cc_library(
    name = "hlo_ops_common",
    srcs = ["lib/Dialect/mhlo/IR/hlo_ops_common.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_common.h"],
    includes = ["include"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

td_library(
    name = "lhlo_gpu_ops_td_files",
    srcs = glob(["include/mlir-hlo/Dialect/lhlo_gpu/IR/*.td"]),
    compatible_with = get_compatible_with_cloud(),
    includes = ["include"],
    deps = [
        ":hlo_ops_td_files",
        ":lhlo_ops_td_files",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "lhlo_gpu_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.td",
    deps = [":lhlo_gpu_ops_td_files"],
)

#TODO(aminim): revisit the naming and grouping of these rules post-move.
gentbl_cc_library(
    name = "canonicalize_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "lib/Dialect/mhlo/IR/",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "lib/Dialect/mhlo/IR/mhlo_canonicalize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/mhlo/IR/mhlo_canonicalize.td",
    deps = [":hlo_ops_td_files"],
)

td_library(
    name = "lhlo_ops_td_files",
    srcs = glob(["include/mlir-hlo/Dialect/lhlo/IR/*.td"]),
    compatible_with = get_compatible_with_cloud(),
    includes = ["include"],
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
    name = "lhlo_structured_interface_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir-hlo/Dialect/lhlo/IR/lhlo_structured_interface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir-hlo/Dialect/lhlo/IR/lhlo_structured_interface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/lhlo/IR/lhlo_structured_interface.td",
    deps = [":lhlo_ops_td_files"],
)

cc_library(
    name = "lhlo_structured_interface",
    srcs = [
        "lib/Dialect/lhlo/IR/lhlo_structured_interface.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/lhlo/IR/lhlo_structured_interface.h",
        "include/mlir-hlo/Dialect/lhlo/IR/lhlo_structured_interface.h.inc",
    ],
    includes = ["include"],
    deps = [
        ":lhlo_structured_interface_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "convert_op_folder",
    srcs = ["lib/utils/convert_op_folder.cc"],
    hdrs = ["include/mlir-hlo/utils/convert_op_folder.h"],
    includes = ["include"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "mlir_hlo",
    srcs = [
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.cc.inc",
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h.inc",
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_attrs.cc.inc",
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_attrs.h.inc",
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_enums.cc.inc",
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_enums.h.inc",
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_typedefs.cc.inc",
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_typedefs.h.inc",
        "lib/Dialect/mhlo/IR/hlo_ops.cc",
        "lib/utils/hlo_utils.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h",
        "include/mlir-hlo/utils/hlo_utils.h",
    ],
    includes = ["include"],
    deps = [
        ":canonicalize_inc_gen",
        ":convert_op_folder",
        ":hlo_ops_attrs_inc_gen",
        ":hlo_ops_common",
        ":hlo_ops_enums_inc_gen",
        ":hlo_ops_inc_gen",
        ":hlo_ops_pattern_gen",
        ":hlo_ops_typedefs_inc_gen",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:base",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:broadcast_utils",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
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
    ],
)

cc_library(
    name = "lhlo",
    srcs = [
        "lib/Dialect/lhlo/IR/lhlo_ops.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h",
        "include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops_structs.h",
        "include/mlir-hlo/utils/lhlo_utils.h",
    ],
    includes = ["include"],
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
    ],
)

cc_library(
    name = "lhlo_gpu",
    srcs = [
        "lib/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h",
    ],
    includes = ["include"],
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
    srcs = [
        "include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.cc.inc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h.inc",
    ],
    includes = ["include"],
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
    srcs = ["lib/Dialect/mhlo/IR/init.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/IR/register.h"],
    deps = [
        ":mlir_hlo",
        # Backward compatibility with the old way of registering CHLO dialect
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
    ],
)

cc_library(
    name = "sink_constants_to_control_flow",
    srcs = [
        "lib/Dialect/mhlo/transforms/sink_constants_to_control_flow.cc",
    ],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "type_conversion",
    srcs = ["lib/Dialect/mhlo/transforms/type_conversion.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/type_conversion.h"],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "map_lmhlo_to_scalar_op",
    hdrs = ["include/mlir-hlo/Dialect/lhlo/transforms/map_lmhlo_to_scalar_op.h"],
    deps = [
        ":lhlo",
        ":map_lhlo_to_hlo_op",
        ":map_mhlo_to_scalar_op",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:SCFDialect",
    ],
)

cc_library(
    name = "map_mhlo_to_scalar_op",
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/map_mhlo_to_scalar_op.h"],
    deps = [
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:VectorDialect",
    ],
)

cc_library(
    name = "map_chlo_to_hlo_op",
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/map_chlo_to_hlo_op.h"],
    deps = [
        ":mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "map_hlo_to_lhlo_op",
    hdrs = ["include/mlir-hlo/Dialect/lhlo/transforms/map_hlo_to_lhlo_op.h"],
    deps = [
        ":lhlo",
        ":mlir_hlo",
    ],
)

cc_library(
    name = "map_lhlo_to_hlo_op",
    hdrs = ["include/mlir-hlo/Dialect/lhlo/transforms/map_lhlo_to_hlo_op.h"],
    deps = [
        ":lhlo",
        ":mlir_hlo",
    ],
)

cc_library(
    name = "lhlo_legalize_to_affine",
    srcs = ["lib/Dialect/lhlo/transforms/lhlo_legalize_to_affine.cc"],
    deps = [
        ":LmhloPassIncGen",
        ":lhlo",
        ":map_lmhlo_to_scalar_op",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "lhlo_legalize_to_parallel_loops",
    srcs = ["lib/Dialect/lhlo/transforms/lhlo_legalize_to_parallel_loops.cc"],
    deps = [
        ":LmhloPassIncGen",
        ":lhlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "codegen_utils",
    srcs = ["lib/utils/codegen_utils.cc"],
    hdrs = ["include/mlir-hlo/utils/codegen_utils.h"],
    includes = ["include"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "placement_utils",
    hdrs = ["include/mlir-hlo/utils/placement_utils.h"],
    includes = ["include"],
    deps = ["@llvm-project//llvm:Support"],
)

cc_library(
    name = "lhlo_elemental_utils",
    srcs = ["lib/Dialect/lhlo/transforms/lhlo_elemental_utils.cc"],
    hdrs = ["include/mlir-hlo/Dialect/lhlo/transforms/lhlo_elemental_utils.h"],
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
    srcs = ["lib/Dialect/mhlo/transforms/legalize_to_linalg_utils.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/legalize_to_linalg_utils.h"],
    deps = [
        ":map_mhlo_to_scalar_op",
        ":mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "sparse_chlo_legalize_to_linalg",
    srcs = ["lib/Dialect/mhlo/transforms/sparse_chlo_legalize_to_linalg.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":legalize_to_linalg_utils",
        ":map_mhlo_to_scalar_op",
        ":mlir_hlo",
        ":type_conversion",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "legalize_to_linalg",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_to_linalg.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/legalize_to_linalg_utils.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":legalize_to_linalg_utils",
        ":map_mhlo_to_scalar_op",
        ":mlir_hlo",
        ":type_conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ArithmeticUtils",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "legalize_mhlo_to_thlo",
    srcs = [
        "lib/Dialect/mhlo/transforms/legalize_mhlo_to_thlo.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
    ],
    includes = ["include"],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        ":thlo",
        ":type_conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "legalize_shape_computations",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_shape_computations.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":map_mhlo_to_scalar_op",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "hlo_legalize_shape_ops_to_standard",
    srcs = ["lib/Dialect/mhlo/transforms/hlo_legalize_shape_ops_to_standard.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        ":type_conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "merge_assuming_ops",
    srcs = ["lib/Dialect/mhlo/transforms/merge_assuming_ops.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":map_chlo_to_hlo_op",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "shape_reification_pass",
    srcs = ["lib/Dialect/mhlo/transforms/shape_reification_pass.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "constraint_fusion_pass",
    srcs = ["lib/Dialect/mhlo/transforms/constraint_fusion_pass.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "group_reduction_dimensions",
    srcs = ["lib/Dialect/mhlo/transforms/group_reduction_dimensions.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "broadcast_propagation",
    srcs = ["lib/Dialect/mhlo/transforms/broadcast_propagation.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "rank_specialization",
    srcs = ["lib/Dialect/mhlo/transforms/rank_specialization.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "lhlo_legalize_to_gpu",
    srcs = ["lib/Dialect/lhlo/transforms/lhlo_legalize_to_gpu.cc"],
    deps = [
        ":LmhloPassIncGen",
        ":lhlo",
        ":map_lmhlo_to_scalar_op",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "lhlo_fuse_linalg",
    srcs = ["lib/Dialect/lhlo/transforms/lhlo_fuse_linalg.cc"],
    hdrs = ["include/mlir-hlo/Dialect/lhlo/transforms/passes.h"],
    deps = [
        ":LmhloPassIncGen",
        ":lhlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgAnalysis",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

cc_library(
    name = "mhlo_canonicalize_reduction",
    srcs = ["lib/Dialect/mhlo/transforms/mhlo_canonicalize_reduction.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "hlo_legalize_to_lhlo",
    srcs = ["lib/Dialect/mhlo/transforms/hlo_legalize_to_lhlo.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":lhlo",
        ":map_hlo_to_lhlo_op",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "hlo_legalize_to_memref",
    srcs = ["lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/bufferizable_op_interface_impl.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":lhlo",
        ":mlir_hlo",
        ":type_conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "hlo_legalize_to_arithmetic",
    srcs = ["lib/Dialect/mhlo/transforms/hlo_legalize_to_arithmetic.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "legalize_to_standard_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "lib/Dialect/mhlo/transforms/",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "lib/Dialect/mhlo/transforms/generated_legalize_to_standard.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/mhlo/transforms/legalize_to_standard_patterns.td",
    deps = [
        ":hlo_ops_td_files",
        "@llvm-project//mlir:ArithmeticOpsTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:MathOpsTdFiles",
    ],
)

cc_library(
    name = "legalize_control_flow",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_control_flow.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "legalize_sort",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_sort.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ArithmeticUtils",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "legalize_to_standard",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_to_standard.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":MhloPassIncGen",
        ":legalize_to_standard_inc_gen",
        ":legalize_trigonometric_to_approximation",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "legalize_einsum_to_dot_general",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_einsum_to_dot_general.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "legalize_gather_to_torch_index_select",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_gather_to_torch_index_select.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "collapse_elementwise_map",
    srcs = ["lib/Dialect/mhlo/transforms/collapse_elementwise_map.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "convert_to_signless_pass",
    srcs = ["lib/Dialect/mhlo/transforms/convert_to_signless_pass.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        ":type_conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "legalize_trigonometric_to_approximation",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_trigonometric_to_approximation.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    includes = ["include"],
    deps = [
        ":MhloPassIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "lower_complex_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "lib/Dialect/mhlo/transforms/",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "lib/Dialect/mhlo/transforms/generated_lower_complex.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/mhlo/transforms/lower_complex_patterns.td",
    deps = [
        ":hlo_ops_td_files",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    #TODO(aminim): find a better name here?
    name = "mhlo_to_mhlo_lowering_patterns",
    srcs = [
        "lib/Dialect/mhlo/transforms/lower_complex.cc",
        "lib/Dialect/mhlo/transforms/lower_general_dot.cc",
        "lib/Dialect/mhlo/transforms/optimize_mhlo.cc",
        "lib/Dialect/mhlo/transforms/sparse_rewriting.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":lower_complex_inc_gen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "materialize_broadcasts",
    srcs = [
        "lib/Dialect/mhlo/transforms/materialize_broadcasts.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "unfuse_batch_norm",
    srcs = ["lib/Dialect/mhlo/transforms/unfuse_batch_norm.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"],
    deps = [
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "legalize_to_tensor_op",
    srcs = ["lib/Dialect/lhlo/transforms/legalize_to_tensor_op.cc"],
    hdrs = ["include/mlir-hlo/Dialect/lhlo/transforms/passes.h"],
    deps = [
        ":LmhloPassIncGen",
        ":lhlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "mhlo_flatten_tuple",
    srcs = ["lib/Dialect/mhlo/transforms/mhlo_flatten_tuple.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "chlo_legalize_to_hlo",
    srcs = ["lib/Dialect/mhlo/transforms/chlo_legalize_to_hlo.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"],
    deps = [
        ":chlo_legalize_to_hlo_inc_gen",
        ":map_chlo_to_hlo_op",
        ":mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:broadcast_utils",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "chlo_legalize_to_hlo_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "lib/Dialect/mhlo/transforms/",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "lib/Dialect/mhlo/transforms/generated_chlo_legalize_to_hlo.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/mhlo/transforms/chlo_legalize_to_hlo_patterns.td",
    deps = [":hlo_ops_td_files"],
)

cc_library(
    name = "chlo_legalize_to_hlo_pass",
    srcs = ["lib/Dialect/mhlo/transforms/chlo_legalize_to_hlo_pass.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":MhloPassIncGen",
        ":chlo_legalize_to_hlo",
        ":mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "expand_hlo_tuples",
    srcs = [
        "lib/Dialect/mhlo/transforms/expand_hlo_tuples.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
    ],
    deps = [
        ":MhloPassIncGen",
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "test_passes",
    srcs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
        "lib/Analysis/test_shape_component_analysis.cc",
        "lib/Analysis/test_userange_analysis.cc",
        "lib/Dialect/mhlo/transforms/materialize_broadcasts_pass.cc",
        "lib/Dialect/mhlo/transforms/optimize_mhlo_pass.cc",
        "lib/Dialect/mhlo/transforms/restrict_max_rank.cc",
        "lib/Dialect/mhlo/transforms/test_infer_shaped_type_pass.cc",
        "lib/Dialect/mhlo/transforms/unfuse_batch_norm_pass.cc",
    ],
    deps = [
        ":MhloPassIncGen",
        ":chlo_legalize_to_hlo",  # build-cleaner: keep
        ":lhlo",
        ":materialize_broadcasts",  # build-cleaner: keep
        ":mhlo_to_mhlo_lowering_patterns",
        ":mlir_hlo",
        ":shape_component_analysis",
        ":sparse_chlo_legalize_to_linalg",  # build-cleaner: keep
        ":transforms_pass_inc_gen",
        ":unfuse_batch_norm",  # build-cleaner: keep
        ":userange_analysis",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "all_passes",
    hdrs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/passes.h",
        "include/mlir-hlo/Dialect/lhlo/transforms/register_passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/register_passes.h",
        "include/mlir-hlo/Transforms/register_passes.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":LmhloPassIncGen",
        ":MhloPassIncGen",
        ":alloc_to_arg_pass",
        ":broadcast_propagation",
        ":buffer_packing",
        ":buffer_reuse",
        ":bufferize",
        ":bufferize_pass",
        ":bufferize_tiled_loop",
        ":chlo_legalize_to_hlo",
        ":chlo_legalize_to_hlo_pass",
        ":collapse_elementwise_map",
        ":collapse_parallel_loops_to_1d_pass",
        ":constraint_fusion_pass",
        ":convert_to_signless_pass",
        ":copy_removal",
        ":expand_hlo_tuples",
        ":generic_host_to_llvm",
        ":gml_st_compose_set_ops",
        ":gml_st_fusion",
        ":gml_st_passes_inc_gen",
        ":gml_st_tiling",
        ":gml_st_to_scf",
        ":gml_st_uncollapse_materialize_ops",
        ":group_reduction_dimensions",
        ":hlo_legalize_shape_ops_to_standard",
        ":hlo_legalize_to_arithmetic",
        ":hlo_legalize_to_lhlo",
        ":hlo_legalize_to_memref",
        ":legalize_control_flow",
        ":legalize_einsum_to_dot_general",
        ":legalize_gather_to_torch_index_select",
        ":legalize_mhlo_to_thlo",
        ":legalize_shape_computations",
        ":legalize_sort",
        ":legalize_to_linalg",
        ":legalize_to_standard",
        ":legalize_to_tensor_op",
        ":legalize_trigonometric_to_approximation",
        ":lhlo",
        ":lhlo_fuse_linalg",
        ":lhlo_legalize_to_affine",
        ":lhlo_legalize_to_gpu",
        ":lhlo_legalize_to_parallel_loops",
        ":lower_index_cast",
        ":merge_assuming_ops",
        ":mhlo_canonicalize_reduction",
        ":mhlo_flatten_tuple",
        ":mhlo_to_mhlo_lowering_patterns",
        ":propagate_static_shapes_to_kernel",
        ":rank_specialization",
        ":shape_reification_pass",
        ":shape_simplification",
        ":sink_constants_to_control_flow",
        ":symbolic_shape_optimization",
        ":test_passes",
        ":tile_loops_pass",
        ":transforms_pass_inc_gen",
        ":unbufferize_pass",
        ":userange_analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "gpu_passes",
    hdrs = [
        "include/mlir-hlo/Transforms/gpu_passes.h",
        "include/mlir-hlo/Transforms/register_gpu_passes.h",
    ],
    deps = [
        ":gpu_fusion_rewrite",
        ":gpu_kernel_lowering_passes",
        ":gpu_transforms_pass_inc_gen",
        ":hlo_to_gpu_pipeline",
        "@llvm-project//mlir:Pass",
    ],
)

gentbl_cc_library(
    name = "gml_st_test_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=GmlStTest",
            ],
            "include/mlir-hlo/Dialect/gml_st/transforms/test_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/gml_st/transforms/test_passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "all_test_passes",
    srcs = ["lib/Dialect/gml_st/transforms/test_passes.cc"],
    hdrs = ["include/mlir-hlo/Dialect/gml_st/transforms/test_passes.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":gml_st_bufferizable_op_interface",
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
    name = "transforms_pass_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=LMHLOTransforms",
            ],
            "include/mlir-hlo/Transforms/passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Transforms/passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "gpu_transforms_pass_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=LMHLOGPUTransforms",
            ],
            "include/mlir-hlo/Transforms/gpu_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Transforms/gpu_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "userange_analysis",
    srcs = ["lib/Analysis/userange_analysis.cc"],
    hdrs = [
        "include/mlir-hlo/Analysis/userange_analysis.h",
    ],
    includes = ["include"],
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
    srcs = ["lib/Analysis/shape_component_analysis.cc"],
    hdrs = [
        "include/mlir-hlo/Analysis/shape_component_analysis.h",
    ],
    includes = ["include"],
    deps = [
        ":mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "generic_host_to_llvm",
    srcs = ["lib/Transforms/generic_host_to_llvm.cc"],
    hdrs = [
        "include/mlir-hlo/Transforms/passes.h",
    ],
    deps = [
        ":mlir_hlo",
        ":transforms_pass_inc_gen",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ArithmeticToLLVM",
        "@llvm-project//mlir:ArithmeticTransforms",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MathToLibm",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorToLLVM",
    ],
)

cc_library(
    name = "buffer_packing",
    srcs = ["lib/Transforms/buffer_packing.cc"],
    hdrs = [
        "include/mlir-hlo/Analysis/userange_analysis.h",
        "include/mlir-hlo/Transforms/passes.h",
    ],
    deps = [
        ":mlir_hlo",
        ":transforms_pass_inc_gen",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "buffer_reuse",
    srcs = ["lib/Transforms/buffer_reuse.cc"],
    hdrs = [
        "include/mlir-hlo/Analysis/userange_analysis.h",
        "include/mlir-hlo/Transforms/passes.h",
    ],
    deps = [
        ":mlir_hlo",
        ":transforms_pass_inc_gen",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "copy_removal",
    srcs = ["lib/Transforms/copy_removal.cc"],
    hdrs = [
        "include/mlir-hlo/Analysis/userange_analysis.h",
        "include/mlir-hlo/Transforms/passes.h",
    ],
    deps = [
        ":lhlo",
        ":mlir_hlo",
        ":transforms_pass_inc_gen",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:CopyOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "gpu_kernel_lowering_passes",
    srcs = [
        "lib/Transforms/gpu_kernel_lowering_passes.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Transforms/gpu_passes.h",
    ],
    deps = [
        ":gpu_transforms_pass_inc_gen",
        ":lhlo",
        ":mlir_hlo",
        "@llvm-project//mlir:ArithmeticToLLVM",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:GPUToROCDLTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "symbolic_shape_optimization",
    srcs = ["lib/Transforms/symbolic_shape_optimization.cc"],
    hdrs = [
        "include/mlir-hlo/Transforms/passes.h",
    ],
    deps = [
        ":mlir_hlo",
        ":shape_component_analysis",
        ":transforms_pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "bufferize",
    srcs = ["lib/Transforms/bufferize.cc"],
    hdrs = ["include/mlir-hlo/Transforms/rewriters.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "bufferize_tiled_loop",
    srcs = ["lib/Dialect/gml_st/transforms/bufferize_tiled_loop.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/passes.h",
        "include/mlir-hlo/Dialect/gml_st/transforms/rewriters.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":gml_st",
        ":gml_st_passes_inc_gen",
        ":lhlo",
        ":mlir_hlo",
        ":type_conversion",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorTransforms",
    ],
)

cc_library(
    name = "bufferize_pass",
    srcs = [
        "lib/Transforms/bufferize_pass.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Transforms/passes.h",
    ],
    deps = [
        ":bufferize",
        ":bufferize_tiled_loop",
        ":gml_st",
        ":gml_st_bufferizable_op_interface",
        ":hlo_legalize_to_memref",
        ":legalize_trigonometric_to_approximation",
        ":lhlo",
        ":mlir_hlo",
        ":transforms_pass_inc_gen",
        ":type_conversion",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ArithmeticTransforms",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorTransforms",
    ],
)

cc_library(
    name = "propagate_static_shapes_to_kernel",
    srcs = [
        "lib/Transforms/propagate_static_shapes_to_kernel.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Transforms/passes.h",
    ],
    deps = [
        ":mlir_hlo",
        ":transforms_pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "lower_index_cast",
    srcs = [
        "lib/Transforms/lower_index_cast_pass.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Transforms/passes.h",
    ],
    deps = [
        ":mlir_hlo",
        ":transforms_pass_inc_gen",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "shape_simplification",
    srcs = [
        "lib/Transforms/shape_simplification.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Transforms/passes.h",
    ],
    deps = [
        ":mlir_hlo",
        ":transforms_pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "tile_loops_pass",
    srcs = [
        "lib/Transforms/tile_loops_pass.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Transforms/passes.h",
    ],
    deps = [
        ":mlir_hlo",
        ":transforms_pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "collapse_parallel_loops_to_1d_pass",
    srcs = [
        "lib/Transforms/collapse_parallel_loops_to_1d_pass.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Transforms/passes.h",
    ],
    deps = [
        ":mlir_hlo",
        ":transforms_pass_inc_gen",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFUtils",
    ],
)

cc_library(
    name = "unbufferize_pass",
    srcs = [
        "lib/Transforms/unbufferize_pass.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Transforms/passes.h",
    ],
    deps = [
        ":mlir_hlo",
        ":transforms_pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "alloc_to_arg_pass",
    srcs = [
        "lib/Transforms/alloc_to_arg_pass.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Transforms/passes.h",
    ],
    deps = [
        ":mlir_hlo",
        ":transforms_pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "hlo_to_gpu_pipeline",
    srcs = [
        "lib/Transforms/hlo_to_gpu_pipeline.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Transforms/gpu_passes.h",
    ],
    deps = [
        ":alloc_to_arg_pass",
        ":bufferize_pass",
        ":collapse_parallel_loops_to_1d_pass",
        ":gml_st_compose_set_ops",
        ":gml_st_fusion",
        ":gml_st_passes",
        ":gml_st_tiling",
        ":gml_st_to_scf",
        ":gpu_kernel_lowering_passes",
        ":legalize_to_linalg",
        ":mlir_hlo",
        ":propagate_static_shapes_to_kernel",
        ":tile_loops_pass",
        ":unbufferize_pass",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithmeticTransforms",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:SCFToGPU",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:ShapeToStandard",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "gpu_fusion_rewrite",
    srcs = [
        "lib/Transforms/gpu_fusion_rewrite.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Transforms/gpu_passes.h",
    ],
    deps = [
        ":gpu_transforms_pass_inc_gen",
        ":hlo_to_gpu_pipeline",
        ":lhlo",
        ":mlir_hlo",
        ":transforms_pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "gml_st_pipeline",
    srcs = ["lib/Transforms/gml_st_pipeline.cc"],
    hdrs = ["include/mlir-hlo/Transforms/gml_st_pipeline.h"],
    deps = [
        ":bufferize_pass",
        ":gml_st_tiling",
        ":gml_st_vectorization",
        ":legalize_mhlo_to_thlo",
        ":legalize_to_linalg",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "gml_st_passes",
    hdrs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/passes.h",
        "include/mlir-hlo/Transforms/register_gml_st_passes.h",
    ],
    deps = [
        ":all_test_passes",
        ":gml_st_passes_inc_gen",
        ":gml_st_pipeline",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
    ],
)

CAPI_HEADERS = [
    "include/mlir-hlo-c/Attributes.h",
    "include/mlir-hlo-c/Dialects.h",
    "include/mlir-hlo-c/Passes.h",
    "include/mlir-hlo-c/Types.h",
]

CAPI_SOURCES = [
    "lib/CAPI/Attributes.cc",
    "lib/CAPI/Dialects.cc",
    "lib/CAPI/Passes.cc",
    "lib/CAPI/Types.cc",
]

cc_library(
    name = "CAPI",
    srcs = CAPI_SOURCES,
    hdrs = CAPI_HEADERS,
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
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "CAPIObjects",
    srcs = CAPI_SOURCES,
    hdrs = CAPI_HEADERS,
    deps = [
        ":all_passes",
        ":mlir_hlo",
        "@llvm-project//mlir:CAPIIRObjects",
    ],
    alwayslink = True,
)

cc_binary(
    name = "mlir-hlo-opt",
    srcs = ["tools/mlir-hlo-opt/mlir-hlo-opt.cc"],
    deps = [
        ":all_passes",
        ":gml_st",
        ":gml_st_passes",
        ":gpu_passes",
        ":hlo_dialect_registration",
        ":lhlo",
        ":lhlo_gpu",
        ":thlo",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:register",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

# Python library.

td_library(
    name = "MhloOpsPyTdFiles",
    srcs = [
        "@llvm-project//mlir:include/mlir/Bindings/Python/Attributes.td",
    ],
    includes = ["include"],
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
            "python/mlir/dialects/_mhlo_ops_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "python/mlir/dialects/MhloOps.td",
    deps = [
        ":MhloOpsPyTdFiles",
    ],
)

filegroup(
    name = "MhloOpsPyFiles",
    srcs = [
        "python/mlir/dialects/mhlo.py",
        ":MhloOpsPyGen",
    ],
)

td_library(
    name = "gml_st_ops_td_files",
    srcs = glob(["include/mlir-hlo/Dialect/gml_st/IR/*.td"]),
    compatible_with = get_compatible_with_cloud(),
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "gml_st_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir-hlo/Dialect/gml_st/IR/gml_st_ops.cc.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "include/mlir-hlo/Dialect/gml_st/IR/gml_st_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "include/mlir-hlo/Dialect/gml_st/IR/gml_st_dialect.cc.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "include/mlir-hlo/Dialect/gml_st/IR/gml_st_types.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "include/mlir-hlo/Dialect/gml_st/IR/gml_st_types.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/gml_st/IR/gml_st_ops.td",
    td_srcs = [
        "include/mlir-hlo/Dialect/gml_st/IR/gml_st_legacy_ops.td",
        "include/mlir-hlo/Dialect/gml_st/IR/gml_st_set_ops.td",
        "include/mlir-hlo/Dialect/gml_st/transforms/compose_set_interface.td",
        "include/mlir-hlo/Dialect/gml_st/transforms/fusion_interface.td",
    ],
    deps = [":gml_st_ops_td_files"],
)

cc_library(
    name = "gml_st",
    srcs = [
        "lib/Dialect/gml_st/IR/gml_st_ops.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h",
    ],
    includes = ["include"],
    deps = [
        ":compose_set_interface",
        ":fusion_interface",
        ":gml_st_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

td_library(
    name = "tiling_interface_td_files",
    srcs = ["include/mlir-hlo/Dialect/gml_st/transforms/tiling_interface.td"],
    includes = ["include"],
    deps = ["@llvm-project//mlir:OpBaseTdFiles"],
)

gentbl_cc_library(
    name = "tiling_interface_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir-hlo/Dialect/gml_st/transforms/tiling_interface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir-hlo/Dialect/gml_st/transforms/tiling_interface.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/gml_st/transforms/tiling_interface.td",
    deps = ["@llvm-project//mlir:OpBaseTdFiles"],
)

cc_library(
    name = "tiling_interface",
    srcs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/tiling_interface.cc.inc",
        "include/mlir-hlo/Dialect/gml_st/transforms/tiling_interface.h.inc",
        "lib/Dialect/gml_st/transforms/tiling_interface.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/tiling_interface.h",
    ],
    includes = ["include"],
    deps = [
        ":tiling_interface_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

cc_library(
    name = "tiling_interface_impl",
    srcs = [
        "lib/Dialect/gml_st/transforms/tiling_interface_impl.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/tiling_interface_impl.h",
    ],
    includes = ["include"],
    deps = [
        ":gml_st",
        ":tiling_interface",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:LinalgUtils",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorUtils",
    ],
)

gentbl_cc_library(
    name = "fusion_interface_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir-hlo/Dialect/gml_st/transforms/fusion_interface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir-hlo/Dialect/gml_st/transforms/fusion_interface.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/gml_st/transforms/fusion_interface.td",
    deps = ["@llvm-project//mlir:OpBaseTdFiles"],
)

cc_library(
    name = "fusion_interface",
    srcs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/fusion_interface.cc.inc",
        "include/mlir-hlo/Dialect/gml_st/transforms/fusion_interface.h.inc",
        "lib/Dialect/gml_st/transforms/fusion_interface.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/fusion_interface.h",
    ],
    includes = ["include"],
    deps = [
        ":fusion_interface_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "fusion_interface_impl",
    srcs = [
        "lib/Dialect/gml_st/transforms/fusion_interface_impl.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/fusion_interface_impl.h",
    ],
    includes = ["include"],
    deps = [
        ":fusion_interface",
        ":gml_st",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:TensorDialect",
    ],
)

gentbl_cc_library(
    name = "compose_set_interface_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir-hlo/Dialect/gml_st/transforms/compose_set_interface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir-hlo/Dialect/gml_st/transforms/compose_set_interface.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/gml_st/transforms/compose_set_interface.td",
    deps = ["@llvm-project//mlir:OpBaseTdFiles"],
)

cc_library(
    name = "compose_set_interface",
    srcs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/compose_set_interface.cc.inc",
        "include/mlir-hlo/Dialect/gml_st/transforms/compose_set_interface.h.inc",
        "lib/Dialect/gml_st/transforms/compose_set_interface.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/compose_set_interface.h",
    ],
    includes = ["include"],
    deps = [
        ":compose_set_interface_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "gml_st_bufferizable_op_interface",
    srcs = ["lib/Dialect/gml_st/transforms/bufferizable_op_interface_impl.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/bufferizable_op_interface_impl.h",
    ],
    includes = ["include"],
    deps = [
        ":gml_st",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

gentbl_cc_library(
    name = "gml_st_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=GmlSt",
            ],
            "include/mlir-hlo/Dialect/gml_st/transforms/passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/gml_st/transforms/passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "gml_st_fusion",
    srcs = ["lib/Dialect/gml_st/transforms/fusion.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/passes.h",
        "include/mlir-hlo/Dialect/gml_st/transforms/rewriters.h",
    ],
    includes = ["include"],
    deps = [
        ":fusion_interface",
        ":fusion_interface_impl",
        ":gml_st",
        ":gml_st_passes_inc_gen",
        ":gml_st_transforms",
        ":thlo",
        ":tiling_interface",
        ":tiling_interface_impl",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "gml_st_tiling",
    srcs = ["lib/Dialect/gml_st/transforms/tiling.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/passes.h",
        "include/mlir-hlo/Dialect/gml_st/transforms/rewriters.h",
    ],
    includes = ["include"],
    deps = [
        ":gml_st",
        ":gml_st_passes_inc_gen",
        ":gml_st_transforms",
        ":tiling_interface",
        ":tiling_interface_impl",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ArithmeticUtils",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "gml_st_compose_set_ops",
    srcs = ["lib/Dialect/gml_st/transforms/compose_set_ops.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/passes.h",
    ],
    includes = ["include"],
    deps = [
        ":compose_set_interface",
        ":gml_st",
        ":gml_st_passes_inc_gen",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "gml_st_uncollapse_materialize_ops",
    srcs = ["lib/Dialect/gml_st/transforms/uncollapse_materialize_ops.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/passes.h",
        "include/mlir-hlo/Dialect/gml_st/transforms/rewriters.h",
    ],
    includes = ["include"],
    deps = [
        ":gml_st",
        ":gml_st_passes_inc_gen",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "gml_st_vectorization",
    srcs = ["lib/Dialect/gml_st/transforms/vectorization.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/passes.h",
    ],
    includes = ["include"],
    deps = [
        ":gml_st",
        ":gml_st_passes_inc_gen",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:VectorDialect",
    ],
)

cc_library(
    name = "gml_st_to_scf",
    srcs = ["lib/Dialect/gml_st/transforms/gml_st_to_scf.cc"],
    hdrs = ["include/mlir-hlo/Dialect/gml_st/transforms/passes.h"],
    includes = ["include"],
    deps = [
        ":gml_st",
        ":gml_st_passes_inc_gen",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "gml_st_transforms",
    srcs = ["lib/Dialect/gml_st/transforms/transforms.cc"],
    hdrs = ["include/mlir-hlo/Dialect/gml_st/transforms/transforms.h"],
    includes = ["include"],
    deps = [
        ":gml_st",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithmeticUtils",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:TensorUtils",
    ],
)

td_library(
    name = "thlo_ops_td_files",
    srcs = glob(["include/mlir-hlo/Dialect/thlo/IR/*.td"]),
    compatible_with = get_compatible_with_cloud(),
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "thlo_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir-hlo/Dialect/thlo/IR/thlo_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir-hlo/Dialect/thlo/IR/thlo_ops.cc.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "include/mlir-hlo/Dialect/thlo/IR/thlo_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "include/mlir-hlo/Dialect/thlo/IR/thlo_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/thlo/IR/thlo_ops.td",
    td_srcs = [
        "include/mlir-hlo/Dialect/gml_st/transforms/fusion_interface.td",
        "include/mlir-hlo/Dialect/gml_st/transforms/tiling_interface.td",
    ],
    deps = [":thlo_ops_td_files"],
)

cc_library(
    name = "thlo",
    srcs = [
        "lib/Dialect/thlo/IR/thlo_ops.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/thlo/IR/thlo_ops.h",
    ],
    includes = ["include"],
    deps = [
        ":fusion_interface",
        ":gml_st",
        ":thlo_ops_inc_gen",
        ":tiling_interface",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ArithmeticUtils",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LinalgUtils",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

# A light-weight runtime support library, used by MLIR code that results
# after lowering some ops in the vector and sparse tensor dialects.
cc_binary(
    name = "libmlir_c_runner_utils.so",
    linkshared = True,
    linkstatic = False,
    deps = [
        "@llvm-project//mlir:mlir_c_runner_utils",
    ],
)
