load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir:__pkg__",
        "//tensorflow/compiler/mlir/tensorflow/transforms:__pkg__",
        "//tensorflow/compiler/mlir/tensorflow/transforms/host_runtime:__pkg__",
        "//tensorflow/compiler/mlir/tf2xla/internal:__pkg__",
    ],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "sparsecore_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=SparseCore",
            ],
            "sparsecore_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "sparsecore_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "sparsecore_passes",
    hdrs = [
        "sparsecore_passes.h",
    ],
    textual_hdrs = [
        "sparsecore_passes.h.inc",
    ],
    deps = [
        ":embedding_pipelining",
        ":embedding_program_key",
        ":embedding_sequencing",
        ":sparsecore_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "embedding_pipelining",
    srcs = ["embedding_pipelining.cc"],
    hdrs = [
        "sparsecore_passes.h",
    ],
    deps = [
        ":sparsecore_passes_inc_gen",
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:attribute_utils",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "embedding_sequencing",
    srcs = ["embedding_sequencing.cc"],
    hdrs = [
        "sparsecore_passes.h",
    ],
    deps = [
        ":sparsecore_passes_inc_gen",
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:attribute_utils",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "embedding_program_key",
    srcs = ["embedding_program_key.cc"],
    hdrs = [
        "sparsecore_passes.h",
    ],
    deps = [
        ":sparsecore_passes_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_xla//xla/mlir_hlo",
    ],
)
