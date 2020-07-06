load("//third_party/mlir:tblgen.bzl", "gentbl")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//third_party/mlir:subpackages"],
    packages = [
        "//babelfish/device/...",
        "//learning/brain/experimental/dtensor/...",
        "//learning/brain/experimental/mlir/...",
        "//learning/brain/google/xla/kernels/...",
        "//learning/brain/google/xla/mlir/...",
        "//learning/pathways/data_parallel/tf2xla/...",
        "//platforms/xla/...",
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/tf2xla/...",
        "//tensorflow/compiler/xla/...",
        "//third_party/iree/...",
        "//third_party/mlir_edge/...",
    ],
)

exports_files(["include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"])

exports_files(["include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.td"])

filegroup(
    name = "hlo_ops_td_files",
    srcs = [
        "include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.td",
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td",
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.td",
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_utils.td",
        "include/mlir-hlo/Dialect/mhlo/IR/infer_fusibility_op_interface.td",
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/LoopLikeInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/ViewLikeInterface.td",
    ],
)

filegroup(
    name = "hlo_ops_base_td",
    srcs = [
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.td",
    ],
)

gentbl(
    name = "chlo_ops_inc_gen",
    strip_include_prefix = "include",
    tbl_outs = [
        ("-gen-op-decls", "include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h.inc"),
        ("-gen-op-defs", "include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.td",
    td_relative_includes = [
        "include",
    ],
    td_srcs = [":hlo_ops_td_files"],
)

gentbl(
    name = "hlo_ops_inc_gen",
    strip_include_prefix = "include",
    tbl_outs = [
        ("-gen-op-decls", "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h.inc"),
        ("-gen-op-defs", "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.cc.inc"),
        ("-gen-struct-attr-decls", "include/mlir-hlo/Dialect/mhlo/IR/hlo_structs.h.inc"),
        ("-gen-struct-attr-defs", "include/mlir-hlo/Dialect/mhlo/IR/hlo_structs.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td",
    td_includes = [
        "include/mlir-hlo/Dialect/mhlo/IR/hlo_utils.td",
    ],
    td_relative_includes = [
        "include",
    ],
    td_srcs = [
        ":hlo_ops_base_td",
        ":hlo_ops_td_files",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
    ],
)

gentbl(
    name = "hlo_ops_base_inc_gen",
    strip_include_prefix = "include",
    tbl_outs = [
        ("-gen-op-decls", "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.h.inc"),
        ("-gen-op-defs", "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.td",
    td_srcs = [":hlo_ops_td_files"],
)

gentbl(
    name = "lhlo_ops_inc_gen",
    strip_include_prefix = "include",
    tbl_outs = [
        ("-gen-op-decls", "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h.inc"),
        ("-gen-op-defs", "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.cc.inc"),
        ("-gen-struct-attr-decls", "include/mlir-hlo/Dialect/mhlo/IR/lhlo_structs.h.inc"),
        ("-gen-struct-attr-defs", "include/mlir-hlo/Dialect/mhlo/IR/lhlo_structs.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.td",
    td_relative_includes = [
        "include",
    ],
    td_srcs = [":hlo_ops_td_files"],
)

gentbl(
    name = "xla_canonicalize_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "lib/Dialect/mhlo/transforms/generated_canonicalize.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/mhlo/transforms/canonicalize.td",
    td_relative_includes = [
        "include",
    ],
    td_srcs = [":hlo_ops_td_files"],
)

gentbl(
    name = "infer_fusibility_op_interface_gen",
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "include/mlir-hlo/Dialect/mhlo/IR/infer_fusibility_op_interface.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "include/mlir-hlo/Dialect/mhlo/IR/infer_fusibility_op_interface.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-hlo/Dialect/mhlo/IR/infer_fusibility_op_interface.td",
    td_relative_includes = [
        "include",
    ],
    td_srcs = [
        ":hlo_ops_td_files",
    ],
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
    deps = [
        ":infer_fusibility_op_interface_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "convert_op_folder",
    srcs = ["lib/utils/convert_op_folder.cc"],
    hdrs = ["include/mlir-hlo/utils/convert_op_folder.h"],
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
        "include/mlir-hlo/utils/broadcast_utils.h",
        "include/mlir-hlo/utils/hlo_utils.h",
    ],
    includes = ["include"],
    deps = [
        ":chlo_ops_inc_gen",
        ":convert_op_folder",
        ":hlo_ops_base_inc_gen",
        ":hlo_ops_inc_gen",
        ":infer_fusibility_op_interface",
        ":xla_canonicalize_inc_gen",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo",
    srcs = [
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.cc.inc",
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h.inc",
        "lib/Dialect/mhlo/IR/lhlo_ops.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h",
    ],
    includes = ["include"],
    deps = [
        ":hlo_ops_base_inc_gen",
        ":lhlo_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
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
    name = "hlo_dialect_registration",
    srcs = ["lib/Dialect/mhlo/IR/dialect_registration.cc"],
    deps = [
        ":hlo",
        ":lhlo",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)
