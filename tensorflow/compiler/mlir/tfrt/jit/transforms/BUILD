load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")
load("//tensorflow/tsl/platform:build_config.bzl", "if_llvm_x86_available")

# TF to TFRT kernels conversion.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/mlir/tfrt:friends"],
    licenses = ["notice"],
)

tfrt_cc_library(
    name = "tf_jitrt_clustering",
    srcs = ["tf_jitrt_clustering.cc"],
    hdrs = ["tf_jitrt_clustering.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/xla/mlir/runtime/utils:constraints",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "tf_jitrt_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
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
        "tf_jitrt_fission.cc",
        "tf_jitrt_fusion.cc",
        "tf_jitrt_legalize_i1_type.cc",
        "tf_jitrt_math_approximation.cc",
        "tf_jitrt_passes.cc",
    ],
    hdrs = ["tf_jitrt_passes.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":tf_jitrt_clustering",
        ":tf_jitrt_passes_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tf2xla/transforms:xla_legalize_tf",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st_passes",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st_transforms",
        "//tensorflow/compiler/xla/mlir_hlo:mhlo_passes",
        "//tensorflow/compiler/xla/mlir_hlo:shape_component_analysis",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:LinalgUtils",
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
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorTransforms",
    ] + if_llvm_x86_available([
        "@llvm-project//mlir:X86VectorTransforms",
    ]),
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
