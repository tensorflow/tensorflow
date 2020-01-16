# Description:
#   MLIR-GPU-specific convolution in XLA service implementation.

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//tensorflow/compiler/xla:friends"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "conv_emitter",
    srcs = ["conv_emitter.cc"],
    hdrs = ["conv_emitter.h"],
    deps = [
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AffineOps",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TransformUtils",
    ],
)

tf_cc_test(
    name = "conv_emitter_test",
    srcs = ["conv_emitter_test.cc"],
    deps = [
        ":conv_emitter",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:test",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AffineDialectRegistration",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardDialectRegistration",
        "@llvm-project//mlir:Transforms",
    ],
)
