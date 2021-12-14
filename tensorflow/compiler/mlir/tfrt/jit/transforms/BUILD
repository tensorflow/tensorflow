load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library")

# TF to TFRT kernels conversion.
package(
    default_visibility = ["//tensorflow/compiler/mlir/tfrt:friends"],
    licenses = ["notice"],
)

tfrt_cc_library(
    name = "clustering",
    srcs = ["clustering.cc"],
    hdrs = ["clustering.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tf_runtime//backends/cpu:cpurt_support",
    ],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TFCPURT",
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
        "buffer_forwarding.cc",
        "clustering_pass.cc",
        "codegen_cwise.cc",
        "codegen_reduction.cc",
        "copy_removal.cc",
        "detensorize_linalg.cc",
        "fission.cc",
        "fuse_fill_into_tiled_reduction.cc",
        "fusion.cc",
        "legalize_i1_type.cc",
        "math_approximation.cc",
        "passes.cc",
        "peel_tiled_loops.cc",
        "sink_unused_outputs.cc",
        "symbolic_shape_optimization.cc",
        "vectorize_tiled_ops.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":clustering",
        ":passes_inc_gen",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:legalize_to_linalg",
        "//tensorflow/compiler/mlir/hlo:shape_component_analysis",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "test_passes_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TFCPURTTest",
            ],
            "test_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "test_passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "test_passes",
    srcs = ["test_passes.cc"],
    hdrs = ["test_passes.h"],
    deps = [
        ":clustering",
        ":test_passes_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)
