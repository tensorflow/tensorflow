load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

td_library(
    name = "mlrt_td_files",
    srcs = [
        "mlrt_dialect.td",
        "mlrt_ops.td",
    ],
    includes = ["."],
    visibility = [
        # copybara:uncomment "//learning/brain/tfrt/mlir:__subpackages__",
        "//learning/infra/mira/distributed:__subpackages__",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "mlrt_ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "mlrt_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "mlrt_ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mlrt_ops.td",
    deps = [":mlrt_td_files"],
)

cc_library(
    name = "mlrt_ops",
    srcs = [
        "mlrt_dialect.cc",
        "mlrt_ops.cc",
    ],
    hdrs = [
        "mlrt_dialect.h",
        "mlrt_ops.h",
    ],
    visibility = [
        # copybara:uncomment "//learning/brain/tfrt/mlir:__subpackages__",
        "//learning/infra/mira/distributed:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
    ],
    deps = [
        ":mlrt_ops_inc_gen",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "tf_mlrt_td_files",
    srcs = [
        "tf_mlrt_dialect.td",
        "tf_mlrt_ops.td",
        "tf_ops.td",
    ],
    includes = ["."],
    visibility = [
        # copybara:uncomment "//learning/infra/mira/distributed:__subpackages__",
    ],
    deps = [
        ":mlrt_td_files",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@tf_runtime//:compiler_td_files",
    ],
)

td_library(
    name = "tf_mlrt_tpu_td_files",
    srcs = [
        "tf_mlrt_tpu_ops.td",
    ],
    includes = ["."],
    visibility = ["//visibility:private"],
    deps = [
        ":mlrt_td_files",
        ":tf_mlrt_td_files",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "tf_mlrt_ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "tf_mlrt_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "tf_mlrt_ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tf_mlrt_ops.td",
    deps = [":tf_mlrt_td_files"],
)

gentbl_cc_library(
    name = "tf_mlrt_tpu_ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "tf_mlrt_tpu_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "tf_mlrt_tpu_ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tf_mlrt_tpu_ops.td",
    deps = [":tf_mlrt_tpu_td_files"],
)

gentbl_cc_library(
    name = "tf_ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "tf_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "tf_ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tf_ops.td",
    deps = [":tf_mlrt_td_files"],
)

cc_library(
    name = "tf_mlrt_ops",
    srcs = ["tf_mlrt_ops.cc"],
    hdrs = ["tf_mlrt_ops.h"],
    visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt/mlrt/application/tensorflow/tests:__subpackages__",
        # copybara:uncomment "//learning/infra/mira/distributed:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
    ],
    deps = [
        ":mlrt_ops",
        ":tf_mlrt_ops_inc_gen",
        ":tf_ops_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_side_effects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:compiler_tfrt_op_interfaces",
        "@tf_runtime//:compiler_tfrt_traits",
    ],
)

cc_library(
    name = "tf_mlrt_tpu_ops",
    srcs = ["tf_mlrt_tpu_ops.cc"],
    hdrs = ["tf_mlrt_tpu_ops.h"],
    visibility = ["//tensorflow/compiler/mlir/tfrt/transforms/mlrt:__subpackages__"],
    deps = [
        ":mlrt_ops",
        ":tf_mlrt_ops",
        ":tf_mlrt_tpu_ops_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
    ],
)
