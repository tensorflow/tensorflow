load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library")

# TF to TFRT kernels conversion.
package(
    default_visibility = ["//tensorflow/compiler/mlir/tfrt:friends"],
    licenses = ["notice"],
)

tfrt_cc_library(
    name = "tf_jitrt_clustering",
    srcs = ["tf_jitrt_clustering.cc"],
    hdrs = ["tf_jitrt_clustering.h"],
    deps = [
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tf_runtime//backends/jitrt:support",
    ],
)

gentbl_cc_library(
    name = "tf_jitrt_passes_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TfJitRt",
            ],
            "tf_jitrt_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tf_jitrt_passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "tf_jitrt_passes",
    srcs = [
        "tf_jitrt_buffer_forwarding.cc",
        "tf_jitrt_clustering_pass.cc",
        "tf_jitrt_copy_removal.cc",
        "tf_jitrt_detensorize_linalg.cc",
        "tf_jitrt_fission.cc",
        "tf_jitrt_fuse_fill_into_tiled_reduction.cc",
        "tf_jitrt_fusion.cc",
        "tf_jitrt_legalize_i1_type.cc",
        "tf_jitrt_lower_vector_transpose.cc",
        "tf_jitrt_math_approximation.cc",
        "tf_jitrt_passes.cc",
        "tf_jitrt_peel_tiled_loops.cc",
        "tf_jitrt_rewrite_vector_multi_reduction.cc",
        "tf_jitrt_symbolic_shape_optimization.cc",
        "tf_jitrt_tile_cwise.cc",
        "tf_jitrt_tile_reduction.cc",
        "tf_jitrt_tile_transpose.cc",
        "tf_jitrt_vectorize_tiled_ops.cc",
    ],
    hdrs = ["tf_jitrt_passes.h"],
    deps = [
        ":tf_jitrt_clustering",
        ":tf_jitrt_passes_inc_gen",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:gml_st",
        "//tensorflow/compiler/mlir/hlo:gml_st_transforms",
        "//tensorflow/compiler/mlir/hlo:legalize_to_linalg",
        "//tensorflow/compiler/mlir/hlo:shape_component_analysis",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:VectorTransforms",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "tf_jitrt_test_passes_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TfJitRtTest",
            ],
            "tf_jitrt_test_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tf_jitrt_test_passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "tf_jitrt_test_passes",
    srcs = ["tf_jitrt_test_passes.cc"],
    hdrs = ["tf_jitrt_test_passes.h"],
    deps = [
        ":tf_jitrt_clustering",
        ":tf_jitrt_test_passes_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)
