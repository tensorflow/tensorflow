load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

licenses(["notice"])

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":friends",
        "//tensorflow:__pkg__",
    ],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/mlir/lite/python/...",
        "//tensorflow/lite/python/...",
        "//tensorflow/python/...",
    ],
)

filegroup(
    name = "api_hdrs",
    srcs = [
        "pybind_api.h",
    ],
    visibility = [":friends"],
)

cc_library(
    name = "pybind_api",
    srcs = ["pybind_api.cc"],
    hdrs = ["pybind_api.h"],
    copts = ["-fexceptions"],  # Required for pybind11.
    features = [
        "-use_header_modules",
        "-parse_headers",
        "-frtti",
    ],
    deps = [
        ":pybind_api_internal",
        "//tensorflow/compiler/mlir/lite:flatbuffer_export",
        "//tensorflow/compiler/mlir/lite:flatbuffer_import",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@pybind11",
    ],
    alwayslink = True,
)

cc_library(
    name = "pybind_api_internal",
    srcs = ["pybind_api_internal.cc"],
    hdrs = ["pybind_api_internal.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_ops",
        "//tensorflow/compiler/mlir/lite/stablehlo:prepare_hlo",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:vhlo_ops",
    ],
)
