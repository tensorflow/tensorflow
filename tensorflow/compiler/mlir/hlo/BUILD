load("//third_party/mlir:tblgen.bzl", "gentbl")

# TODO(b/160617323): Decouple MLIR HLO from TensorFlow/XLA
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

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

#TODO(aminim): revisit the naming and grouping of these rules post-move.
gentbl(
    name = "canonicalize_inc_gen",
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
        ":canonicalize_inc_gen",
        ":chlo_ops_inc_gen",
        ":convert_op_folder",
        ":hlo_ops_base_inc_gen",
        ":hlo_ops_inc_gen",
        ":infer_fusibility_op_interface",
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

cc_library(
    name = "sink_constants_to_control_flow",
    srcs = ["lib/Dialect/mhlo/transforms/sink_constants_to_control_flow.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "map_lmhlo_to_scalar_op",
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/map_lmhlo_to_scalar_op.h"],
    deps = [
        ":hlo",
        ":lhlo",
        ":map_hlo_to_lhlo_op",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:StandardOps",
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
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_legalize_to_parallel_loops",
    srcs = ["lib/Dialect/mhlo/transforms/lhlo_legalize_to_parallel_loops.cc"],
    deps = [
        ":lhlo",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_legalize_to_llvm",
    srcs = ["lib/Dialect/mhlo/transforms/lhlo_legalize_to_llvm.cc"],
    deps = [
        ":lhlo",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_to_linalg",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_to_linalg.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"],
    deps = [
        ":hlo",
        ":lhlo",
        ":map_lmhlo_to_scalar_op",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "transform_unranked_hlo",
    srcs = ["lib/Dialect/mhlo/transforms/transform_unranked_hlo.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"],
    deps = [
        ":hlo",
        "@com_google_absl//absl/memory",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
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
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
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
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_copy_removal",
    srcs = ["lib/Dialect/mhlo/transforms/lhlo_copy_removal.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":lhlo",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
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
        "@com_google_absl//absl/memory",
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
    name = "cycle_detector",
    srcs = ["lib/utils/cycle_detector.cc"],
    hdrs = ["include/mlir-hlo/utils/cycle_detector.h"],
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

gentbl(
    name = "legalize_to_standard_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "lib/Dialect/mhlo/transforms/generated_legalize_to_standard.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/mhlo/transforms/legalize_to_standard_patterns.td",
    td_relative_includes = [
        "../hlo/include",
    ],
    td_srcs = [
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
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_to_standard",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_to_standard.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/passes.h"],
    deps = [
        ":hlo",
        ":legalize_tanh_to_approximation",
        ":legalize_to_standard_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_tanh_to_approximation",
    srcs = ["lib/Dialect/mhlo/transforms/legalize_tanh_to_approximation.cc"],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

gentbl(
    name = "lower_complex_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "lib/Dialect/mhlo/transforms/generated_lower_complex.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/mhlo/transforms/lower_complex_patterns.td",
    td_relative_includes = [
        "../hlo/include",
    ],
    td_srcs = [
        ":hlo_ops_td_files",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

cc_library(
    #TODO(aminim): find a better name here?
    name = "mhlo_to_mhlo_lowering_patterns",
    srcs = [
        "lib/Dialect/mhlo/transforms/generated_lower_complex.inc",
        "lib/Dialect/mhlo/transforms/lower_complex.cc",
        "lib/Dialect/mhlo/transforms/lower_general_dot.cc",
    ],
    hdrs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/passes.h",
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
    ],
    deps = [
        ":hlo",
        ":hlo_dialect_registration",
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
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "chlo_legalize_to_hlo",
    srcs = ["lib/Dialect/mhlo/transforms/chlo_legalize_to_hlo.cc"],
    hdrs = ["include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"],
    deps = [
        ":hlo",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "test_passes",
    srcs = [
        "include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h",
        "lib/Dialect/mhlo/transforms/chlo_legalize_to_hlo_pass.cc",
        "lib/Dialect/mhlo/transforms/lhlo_legalize_to_llvm_pass.cc",
        "lib/Dialect/mhlo/transforms/materialize_broadcasts_pass.cc",
        "lib/Dialect/mhlo/transforms/test_infer_shaped_type_pass.cc",
        "lib/Dialect/mhlo/transforms/unfuse_batch_norm_pass.cc",
    ],
    deps = [
        ":chlo_legalize_to_hlo",  # build-cleaner: keep
        ":hlo",
        ":lhlo",
        ":lhlo_legalize_to_llvm",  # build-cleaner: keep
        ":materialize_broadcasts",  # build-cleaner: keep
        ":unfuse_batch_norm",  # build-cleaner: keep
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:CFGTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "all_passes_for_testing",
    visibility = [
        "//tensorflow/compiler/mlir:__subpackages__",
    ],
    deps = [
        ":chlo_legalize_to_hlo",
        ":hlo_dialect_registration",
        ":hlo_legalize_to_lhlo",
        ":legalize_control_flow",
        ":legalize_tanh_to_approximation",
        ":legalize_to_linalg",
        ":legalize_to_standard",
        ":lhlo",
        ":lhlo_copy_removal",
        ":lhlo_fuse_linalg",
        ":lhlo_legalize_to_affine",
        ":lhlo_legalize_to_gpu",
        ":lhlo_legalize_to_parallel_loops",
        ":mhlo_fusion",
        ":mhlo_to_mhlo_lowering_patterns",
        ":sink_constants_to_control_flow",
        ":test_passes",
        ":transform_unranked_hlo",
    ],
)

cc_binary(
    name = "mlir-hlo-opt",
    deps = [
        ":all_passes_for_testing",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:MlirOptMain",
    ],
)
