load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "//tensorflow:tensorflow.default.bzl",
    "get_compatible_with_portable",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:__subpackages__",
        "//tensorflow/compiler/mlir/quantization/tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "component",
    hdrs = ["component.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "io",
    srcs = ["io.cc"],
    hdrs = ["io.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:env",
    ],
)

tf_cc_test(
    name = "io_test",
    srcs = ["io_test.cc"],
    deps = [
        ":io",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "graph_def",
    srcs = [],
    hdrs = ["graph_def.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "graph_def_test",
    srcs = ["graph_def_test.cc"],
    deps = [
        ":graph_def",
        "//tensorflow/core:protos_all_cc",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "debugger",
    srcs = ["debugger.cc"],
    hdrs = ["debugger.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:graph_def",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "export",
    srcs = [],
    hdrs = ["export.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/strings:string_view",
    ],
)
