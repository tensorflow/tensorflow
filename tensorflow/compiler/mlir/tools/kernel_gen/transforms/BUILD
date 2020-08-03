load("//third_party/mlir:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//tensorflow/compiler/mlir/tools/kernel_gen:friends"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tf_framework_legalize_to_llvm",
    srcs = ["tf_framework_legalize_to_llvm.cc"],
    hdrs = ["rewriters.h"],
    deps = [
        "//tensorflow/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl(
    name = "tf_framework_passes_inc_gen",
    tbl_outs = [("-gen-pass-decls", "tf_framework_passes.h.inc")],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    td_srcs = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    srcs = [
        "register_passes.cc",
        "tf_framework_legalize_to_llvm_pass.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":tf_framework_legalize_to_llvm",
        ":tf_framework_passes_inc_gen",
        "//tensorflow/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
    ],
    alwayslink = 1,
)
