load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = [
        "//tensorflow:internal",
        "@tf_runtime//:friends",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "async_runtime_api",
    srcs = ["async_runtime_api.cc"],
    hdrs = ["async_runtime_api.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/xla/runtime:async_runtime",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:mlir_async_runtime_api",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "constraints",
    srcs = ["constraints.cc"],
    hdrs = ["constraints.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/xla/runtime:constraints",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@tf_runtime//:support",
    ],
)
