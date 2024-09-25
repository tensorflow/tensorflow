# Placeholder: load py_proto_library
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/compiler/mlir/lite:__subpackages__",
        "//tensorflow/lite/python/converter:__subpackages__",
        "//tensorflow/lite/toco/python:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "error_collector",
    srcs = ["error_collector.cc"],
    hdrs = ["error_collector.h"],
    deps = [
        ":converter_error_data_proto_cc",
        ":types_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "error_collector_inst",
    srcs = ["error_collector_inst.cc"],
    hdrs = ["error_collector_inst.h"],
    deps = [
        ":converter_error_data_proto_cc",
        ":error_collector",
        ":types_util",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "error_collector_inst_test",
    srcs = ["error_collector_inst_test.cc"],
    data = [
        "testdata/strided_slice.mlir",
    ],
    deps = [
        ":converter_error_data_proto_cc",
        ":error_collector",
        ":error_collector_inst",
        ":types_util",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:resource_loader",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "types_util",
    srcs = ["types_util.cc"],
    hdrs = ["types_util.h"],
    deps = [
        ":converter_error_data_proto_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_proto_library(
    name = "converter_error_data_proto",
    srcs = ["converter_error_data.proto"],
    visibility = [
        "//visibility:public",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "converter_error_data_proto_py",
#     api_version = 2,
#     visibility = [
#         "//visibility:public",
#     ],
#     deps = [":converter_error_data_proto"],
# )
# copybara:uncomment_end
