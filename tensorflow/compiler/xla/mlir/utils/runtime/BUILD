load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
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
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/xla/runtime:async_runtime",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:mlir_async_runtime_api",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "c_runner_utils",
    hdrs = ["c_runner_utils.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//mlir:mlir_c_runner_utils",
    ],
)

cc_library(
    name = "constraints",
    srcs = ["constraints.cc"],
    hdrs = ["constraints.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/xla/runtime:constraints",
        "//tensorflow/compiler/xla/runtime:errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
