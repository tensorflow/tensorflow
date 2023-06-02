load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_cloud")
load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/xla/mlir/runtime:friends"],
    licenses = ["notice"],
)

cc_library(
    name = "async_runtime_api",
    srcs = ["async_runtime_api.cc"],
    hdrs = ["async_runtime_api.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/xla/runtime:async_runtime",
        "//tensorflow/tsl/platform:platform_port",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/functional:any_invocable",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//mlir:mlir_async_runtime_api",
        "@tf_runtime//:async_value",
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

cc_library(
    name = "custom_calls",
    srcs = ["custom_calls.cc"],
    hdrs = ["custom_calls.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "float_16bits",
    hdrs = ["float_16bits.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//mlir:mlir_float16_utils",
    ],
)
