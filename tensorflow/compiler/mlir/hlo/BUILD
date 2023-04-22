load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")

# TODO(b/160617323): Decouple MLIR HLO from TensorFlow/XLA
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["@llvm-project//mlir:subpackages"],
    packages = [
        "//babelfish/device/...",
        "//learning/brain/experimental/dtensor/...",
        "//learning/brain/experimental/mlir/...",
        "//learning/brain/google/xla/kernels/...",
        "//learning/brain/google/xla/mlir/...",
        "//learning/deepmind/partir/...",
        "//learning/pathways/data_parallel/tf2xla/...",
        "//platforms/xla/...",
        "//third_party/iree/...",
        "//third_party/mlir_edge/...",
        "//third_party/py/mlir/...",
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/tf2xla/...",
        "//tensorflow/compiler/xla/...",
        # Allow visibility from the mlir language server.
        "//learning/brain/mlir/mlir_lsp_server/...",
    ],
)

exports_files([
    "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td",
    "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.td",
])

td_library(
    name = "hlo_ops_td_files",
    srcs = glob(["include/mlir-hlo/Dialect/mhlo/IR/*.td"]) + [
        # TODO(gcmn): These should be encapsulate in a td_library.
        "@llvm-project//mlir:include/mlir/Interfaces/ControlFlowInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/CopyOpInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/LoopLikeInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/ViewLikeInterface.td",
        "@llvm-project//mlir:include/mlir/Dialect/Shape/IR/ShapeBase.td",
        "@llvm-project//mlir:include/mlir/Dialect/Shape/IR/ShapeOps.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:MemRefOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
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
            "include/mlir-hlo/Dialect/mhlo/transforms/lmhlo_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/transforms/lmhlo_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "chlo_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.td",
    deps = [":hlo_ops_td_files"],
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
    name = "hlo_ops_base_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.td",
    deps = [":hlo_ops_td_files"],
)

gentbl_cc_library(
    name = "hlo_ops_base_structs_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-struct-attr-decls"],
            "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.h.inc",
        ),
        (
            ["-gen-struct-attr-defs"],
            "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.td",
    deps = [":hlo_ops_td_files"],
)

gentbl_cc_library(
    name = "hlo_ops_base_enums_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_enums.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.td",
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
        "@llvm-project//mlir:StdOpsTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "lhlo_ops_structs_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-struct-attr-decls"],
            "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops_structs.h.inc",
        ),
        (
            ["-gen-struct-attr-defs"],
            "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops_structs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops_structs.td",
    deps = [":hlo_ops_td_files"],
)

gentbl_cc_library(
    name = "lhlo_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.td",
    deps = [":hlo_ops_td_files"],
)

gentbl_cc_library(
    name = "lhlo_gpu_ops_structs_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-struct-attr-decls"],
            "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops_structs.h.inc",
        ),
        (
            ["-gen-struct-attr-defs"],
            "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops_structs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops_structs.td",
    deps = [":hlo_ops_td_files"],
)

gentbl_cc_library(
    name = "lhlo_gpu_ops_enums_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops_enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops_enums.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops_enums.td",
    deps = [":hlo_ops_td_files"],
)

gentbl_filegroup(
    name = "hlo_ops_doc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
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
            ["-gen-dialect-doc"],
            "g3doc/lhlo_ops.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.td",
    deps = [":hlo_ops_td_files"],
)

cc_library(
    name = "hlo_ops_common",
    srcs = ["lib/Dialect/mhlo/IR/hlo_ops_common.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_common.h"],
    includes = ["include"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "lhlo_gpu_ops_structs",
    srcs = [
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops_structs.cc.inc",
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops_structs.h.inc",
        "lib/Dialect/mhlo/IR/lhlo_gpu_ops_structs.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops_structs.h",
    ],
    includes = ["include"],
    deps = [
        ":lhlo_gpu_ops_structs_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "lhlo_gpu_ops_enums",
    srcs = [
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops_enums.cc.inc",
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops_enums.h.inc",
        "lib/Dialect/mhlo/IR/lhlo_gpu_ops_enums.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops_enums.h",
    ],
    includes = ["include"],
    deps = [
        ":lhlo_gpu_ops_enums_inc_gen",
        "@llvm-project//llvm:Support",
    ],
)

gentbl_cc_library(
    name = "lhlo_gpu_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops.td",
    deps = [":hlo_ops_td_files"],
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

gentbl_cc_library(
    name = "infer_fusibility_op_interface_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir-hlo/Dialect/mhlo/IR/infer_fusibility_op_interface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir-hlo/Dialect/mhlo/IR/infer_fusibility_op_interface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/infer_fusibility_op_interface.td",
    deps = [":hlo_ops_td_files"],
)

cc_library(
    name = "infer_fusibility_op_interface",
    srcs = [
        "lib/Dialect/mhlo/IR/infer_fusibility_op_interface.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/IR/infer_fusibility_op_interface.h",
        "include/mlir-hlo/Dialect/mhlo/IR/infer_fusibility_op_interface.h.inc",
    ],
    includes = ["include"],
    deps = [
        ":infer_fusibility_op_interface_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "lhlo_structured_interface_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/mlir-hlo/Dialect/mhlo/IR/lhlo_structured_interface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/mlir-hlo/Dialect/mhlo/IR/lhlo_structured_interface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/lhlo_structured_interface.td",
    deps = [":hlo_ops_td_files"],
)

cc_library(
    name = "lhlo_structured_interface",
    srcs = [
        "lib/Dialect/mhlo/IR/lhlo_structured_interface.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_structured_interface.h",
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_structured_interface.h.inc",
    ],
    includes = ["include"],
    deps = [
        ":lhlo_structured_interface_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "hlo_ops_base_structs",
    srcs = [
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.h.inc",
        "lib/Dialect/mhlo/IR/hlo_ops_base_structs.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.h",
    ],
    includes = ["include"],
    deps = [
        ":hlo_ops_base_structs_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "hlo_ops_base_enums",
    srcs = [
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_enums.h.inc",
        "lib/Dialect/mhlo/IR/hlo_ops_base_enums.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_enums.h",
    ],
    includes = ["include"],
    deps = [
        ":hlo_ops_base_enums_inc_gen",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "convert_op_folder",
    srcs = ["lib/utils/convert_op_folder.cc"],
    hdrs = ["include/mlir-hlo/utils/convert_op_folder.h"],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "hlo",
    srcs = [
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.cc.inc",
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h.inc",
        "lib/Dialect/mhlo/IR/chlo_ops.cc",
        "lib/Dialect/mhlo/IR/hlo_ops.cc",
        "lib/utils/broadcast_utils.cc",
        "lib/utils/hlo_utils.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h",
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h",
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.h",
        "include/mlir-hlo/utils/broadcast_utils.h",
        "include/mlir-hlo/utils/hlo_utils.h",
    ],
    includes = ["include"],
    deps = [
        ":canonicalize_inc_gen",
        ":chlo_ops_inc_gen",
        ":convert_op_folder",
        ":hlo_ops_base_enums",
        ":hlo_ops_base_inc_gen",
        ":hlo_ops_base_structs",
        ":hlo_ops_common",
        ":hlo_ops_inc_gen",
        ":hlo_ops_pattern_gen",
        ":infer_fusibility_op_interface",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo",
    srcs = [
        "lib/Dialect/mhlo/IR/lhlo_ops.cc",
        "lib/Dialect/mhlo/IR/lhlo_ops_structs.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h",
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops_structs.h",
        "include/mlir-hlo/utils/lhlo_utils.h",
    ],
    includes = ["include"],
    deps = [
        ":hlo_ops_base_enums",
        ":hlo_ops_base_inc_gen",
        ":hlo_ops_base_structs",
        ":hlo_ops_common",
        ":lhlo_ops_inc_gen",
        ":lhlo_ops_structs_inc_gen",
        ":lhlo_structured_interface",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:CopyOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_gpu",
    srcs = [
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops.cc.inc",
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops.h.inc",
        "lib/Dialect/mhlo/IR/lhlo_gpu_ops.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops.h",
    ],
    includes = ["include"],
    deps = [
        ":hlo",
        ":hlo_ops_base_enums",
        ":hlo_ops_base_structs",
        ":hlo_ops_common",
        ":infer_fusibility_op_interface",
        ":lhlo",
        ":lhlo_gpu_ops_enums",
        ":lhlo_gpu_ops_inc_gen",
        ":lhlo_gpu_ops_structs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:CopyOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "DiscRalPassIncGen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=RAL",
            ],
            "include/mlir-hlo/Dialect/mhlo/transforms/disc_ral_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/transforms/disc_ral_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "disc_ral_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.td",
    deps = [":hlo_ops_td_files"],
)

cc_library(
    name = "disc_ral",
    srcs = [
        "include/mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.cc.inc",
        "include/mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.h.inc",
        "lib/Dialect/mhlo/IR/disc_ral_ops.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.h",
    ],
    includes = ["include"],
    deps = [
        ":disc_ral_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:CopyOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ral_inject_execution_context",
    srcs = ["lib/Dialect/mhlo/transforms/ral_inject_execution_context.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":disc_ral",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ral_lower_to_library_call",
    srcs = ["lib/Dialect/mhlo/transforms/ral_lower_to_library_call.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":disc_ral",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ral_legalize_to_llvm",
    srcs = ["lib/Dialect/mhlo/transforms/ral_legalize_to_llvm.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":disc_ral",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToGPURuntimeTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hlo_dialect_registration",
    srcs = ["lib/Dialect/mhlo/IR/init.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/IR/register.h"],
    deps = [
        ":disc_ral",
        ":hlo",
        ":lhlo",
        ":lhlo_gpu",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "sink_constants_to_control_flow",
    srcs = [
        "lib/Dialect/mhlo/transforms/sink_constants_to_control_flow.cc",
    ],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":hlo",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mhlo_control_flow_to_scf",
    srcs = ["lib/Dialect/mhlo/transforms/mhlo_control_flow_to_scf.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":hlo",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "mhlo_mark_shape_calc",
    srcs = ["lib/Dialect/mhlo/transforms/mhlo_mark_shape_calc.cc"],
    deps = [
        ":hlo",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
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
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "map_lmhlo_to_scalar_op",
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/map_lmhlo_to_scalar_op.h"],
    deps = [
        ":hlo",
        ":lhlo",
        ":map_hlo_to_lhlo_op",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
    ],
)

cc_library(
    name = "map_chlo_to_hlo_op",
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/map_chlo_to_hlo_op.h"],
    deps = [
        ":hlo",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "map_hlo_to_lhlo_op",
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/map_hlo_to_lhlo_op.h"],
    deps = [
        ":hlo",
        ":lhlo",
    ],
)

cc_library(
    name = "lhlo_legalize_to_affine",
    srcs = ["lib/Dialect/mhlo/transforms/lhlo_legalize_to_affine.cc"],
    deps = [
        ":hlo",
        ":lhlo",
        ":map_lmhlo_to_scalar_op",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_legalize_to_parallel_loops",
    srcs = ["lib/Dialect/mhlo/transforms/lhlo_legalize_to_parallel_loops.cc"],
    deps = [
        ":lhlo",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "codegen_utils",
    srcs = ["lib/utils/codegen_utils.cc"],
    hdrs = ["include/mlir-hlo/utils/codegen_utils.h"],
    includes = ["include"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "placement_utils",
    hdrs = ["include/mlir-hlo/utils/placement_utils.h"],
    includes = ["include"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "lhlo_elemental_utils",
    srcs = ["lib/Dialect/mhlo/transforms/lhlo_elemental_utils.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/lhlo_elemental_utils.h"],
    deps = [
        ":codegen_utils",
        ":hlo",
        ":lhlo",
        ":map_lmhlo_to_scalar_op",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "fusion_utils",
    srcs = ["lib/Dialect/mhlo/transforms/fusion_utils.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/fusion_utils.h"],
    deps = [
        ":lhlo",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "disc_supported_list",
    hdrs = ["include/mlir-hlo/utils/disc_supported_list.h.inc"],
)

cc_library(
    name = "lhlo_legalize_roots_to_loops",
    srcs = ["lib/Dialect/mhlo/transforms/lhlo_legalize_roots_to_loops.cc"],
    deps = [
        ":LmhloPassIncGen",
        ":codegen_utils",
        ":disc_supported_list",
        ":fusion_utils",
        ":hlo",
        ":lhlo",
        ":lhlo_elemental_utils",
        ":map_lmhlo_to_scalar_op",
        ":placement_utils",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "input_inline_fusion",
    srcs = ["lib/Dialect/mhlo/transforms/input_inline_fusion_pass.cc"],
    deps = [
        ":LmhloPassIncGen",
        ":disc_supported_list",
        ":hlo",
        ":lhlo",
        ":lhlo_elemental_utils",
        ":map_lmhlo_to_scalar_op",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_to_linalg",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_to_linalg.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":hlo",
        ":lhlo",
        ":map_lmhlo_to_scalar_op",
        ":pass_details",
        ":type_conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "broadcast_propagation",
    srcs = ["lib/Dialect/mhlo/transforms/broadcast_propagation.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":hlo",
        ":map_chlo_to_hlo_op",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rank_specialization",
    srcs = ["lib/Dialect/mhlo/transforms/rank_specialization.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":hlo",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_legalize_to_gpu",
    srcs = ["lib/Dialect/mhlo/transforms/lhlo_legalize_to_gpu.cc"],
    deps = [
        ":hlo",
        ":lhlo",
        ":map_lmhlo_to_scalar_op",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_fuse_linalg",
    srcs = ["lib/Dialect/mhlo/transforms/lhlo_fuse_linalg.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":lhlo",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mhlo_canonicalize_reduction",
    srcs = ["lib/Dialect/mhlo/transforms/mhlo_canonicalize_reduction.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":hlo",
        ":pass_details",
        "@llvm-project//llvm:Support",
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
        ":hlo",
        ":lhlo",
        ":map_hlo_to_lhlo_op",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hlo_legalize_to_memref",
    srcs = ["lib/Dialect/mhlo/transforms/hlo_legalize_to_memref.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":hlo",
        ":pass_details",
        ":type_conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "cycle_detector",
    srcs = ["lib/utils/cycle_detector.cc"],
    hdrs = ["include/mlir-hlo/utils/cycle_detector.h"],
    includes = ["include"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "cycle_detector_test",
    srcs = ["lib/utils/cycle_detector_test.cc"],
    deps = [
        ":cycle_detector",
        # TODO(b/160617323): Decouple MLIR HLO from TensorFlow/XLA
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "mhlo_fusion",
    srcs = ["lib/Dialect/mhlo/transforms/mhlo_fusion.cc"],
    deps = [
        ":cycle_detector",
        ":hlo",
        ":pass_details",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
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
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

cc_library(
    name = "legalize_control_flow",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_control_flow.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":hlo",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_to_standard",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_to_standard.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":hlo",
        ":legalize_to_standard_inc_gen",
        ":legalize_trigonometric_to_approximation",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_einsum_to_dot_general",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_einsum_to_dot_general.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":hlo",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_gather_to_torch_index_select",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_gather_to_torch_index_select.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":hlo",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
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
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
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
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

cc_library(
    #TODO(aminim): find a better name here?
    name = "mhlo_to_mhlo_lowering_patterns",
    srcs = [
        "lib/Dialect/mhlo/transforms/lower_complex.cc",
        "lib/Dialect/mhlo/transforms/lower_general_dot.cc",
        "lib/Dialect/mhlo/transforms/optimize_mhlo.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":hlo",
        ":lower_complex_inc_gen",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
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
        ":hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "unfuse_batch_norm",
    srcs = ["lib/Dialect/mhlo/transforms/unfuse_batch_norm.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
    ],
    deps = [
        ":hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "legalize_tensor_load_op",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_tensor_load_op.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":lhlo",
        ":pass_details",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "lhlo_fusion",
    srcs = ["lib/Dialect/mhlo/transforms/lhlo_fusion.cc"],
    deps = [
        ":cycle_detector",
        ":fusion_utils",
        ":lhlo",
        ":pass_details",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_fusion_inliner",
    srcs = ["lib/Dialect/mhlo/transforms/lhlo_fusion_inliner.cc"],
    deps = [
        ":lhlo",
        ":pass_details",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "chlo_legalize_to_hlo",
    srcs = ["lib/Dialect/mhlo/transforms/chlo_legalize_to_hlo.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"],
    deps = [
        ":chlo_legalize_to_hlo_inc_gen",
        ":hlo",
        ":map_chlo_to_hlo_op",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
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
    name = "pass_details",
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/PassDetail.h",
    ],
    visibility = [
        "//visibility:private",  # This target is a private detail of pass implementations
    ],
    deps = [
        ":DiscRalPassIncGen",
        ":LmhloPassIncGen",
        ":MhloPassIncGen",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "test_passes",
    srcs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
        "lib/Analysis/test_userange_analysis.cc",
        "lib/Dialect/mhlo/transforms/chlo_legalize_to_hlo_pass.cc",
        "lib/Dialect/mhlo/transforms/materialize_broadcasts_pass.cc",
        "lib/Dialect/mhlo/transforms/optimize_mhlo_pass.cc",
        "lib/Dialect/mhlo/transforms/test_infer_shaped_type_pass.cc",
        "lib/Dialect/mhlo/transforms/unfuse_batch_norm_pass.cc",
    ],
    deps = [
        ":chlo_legalize_to_hlo",  # build-cleaner: keep
        ":hlo",
        ":lhlo",
        ":materialize_broadcasts",  # build-cleaner: keep
        ":pass_details",
        ":transforms_pass_details",
        ":unfuse_batch_norm",  # build-cleaner: keep
        ":userange_analysis",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "all_passes",
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/register_passes.h",
        "include/mlir-hlo/Transforms/register_passes.h",
    ],
    visibility = [
        ":friends",
    ],
    deps = [
        ":DiscRalPassIncGen",
        ":LmhloPassIncGen",
        ":MhloPassIncGen",
        ":broadcast_propagation",
        ":buffer_reuse",
        ":chlo_legalize_to_hlo",
        ":hlo_legalize_to_lhlo",
        ":hlo_legalize_to_memref",
        ":input_inline_fusion",
        ":legalize_control_flow",
        ":legalize_einsum_to_dot_general",
        ":legalize_gather_to_torch_index_select",
        ":legalize_tensor_load_op",
        ":legalize_to_linalg",
        ":legalize_to_standard",
        ":legalize_trigonometric_to_approximation",
        ":lhlo",
        ":lhlo_fuse_linalg",
        ":lhlo_fusion",
        ":lhlo_fusion_inliner",
        ":lhlo_legalize_roots_to_loops",
        ":lhlo_legalize_to_affine",
        ":lhlo_legalize_to_gpu",
        ":lhlo_legalize_to_parallel_loops",
        ":mhlo_canonicalize_reduction",
        ":mhlo_control_flow_to_scf",
        ":mhlo_fusion",
        ":mhlo_mark_shape_calc",
        ":mhlo_to_mhlo_lowering_patterns",
        ":ral_inject_execution_context",
        ":ral_legalize_to_llvm",
        ":ral_lower_to_library_call",
        ":rank_specialization",
        ":sink_constants_to_control_flow",
        ":test_passes",
        ":transforms_pass_details",
        ":transforms_pass_inc_gen",
        ":userange_analysis",
        "@llvm-project//mlir:Pass",
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

cc_library(
    name = "transforms_pass_details",
    hdrs = [
        "include/mlir-hlo/Transforms/PassDetail.h",
        "include/mlir-hlo/Transforms/passes.h",
    ],
    visibility = [
        "//visibility:private",  # This target is a private detail of pass implementations
    ],
    deps = [
        ":transforms_pass_inc_gen",
        "@llvm-project//mlir:Pass",
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
        ":hlo",
        ":transforms_pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "buffer_reuse",
    srcs = ["lib/Transforms/buffer_reuse.cc"],
    hdrs = [
        "include/mlir-hlo/Analysis/userange_analysis.h",
        "include/mlir-hlo/Transforms/PassDetail.h",
        "include/mlir-hlo/Transforms/passes.h",
    ],
    deps = [
        ":hlo",
        ":transforms_pass_inc_gen",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_binary(
    name = "mlir-hlo-opt",
    srcs = [
        "tools/mlir-hlo-opt/mlir-hlo-opt.cpp",
    ],
    deps = [
        ":all_passes",
        ":disc_ral",
        ":hlo",
        ":lhlo",
        ":lhlo_gpu",
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
            "python/_mhlo_ops_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "python/MhloOps.td",
    deps = [
        ":MhloOpsPyTdFiles",
    ],
)

filegroup(
    name = "MhloOpsPyFiles",
    srcs = [
        "python/mhlo.py",
        ":MhloOpsPyGen",
    ],
)
