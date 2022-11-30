load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/lite:__subpackages__",
        "//tensorflow/lite/python:__subpackages__",
        "//tensorflow/lite/toco/python:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "error_collector",
    srcs = ["error_collector.cc"],
    hdrs = ["error_collector.h"],
    deps = [
        ":types_util",
        "//tensorflow/lite/python/metrics:converter_error_data_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "error_collector_inst",
    srcs = ["error_collector_inst.cc"],
    hdrs = ["error_collector_inst.h"],
    deps = [
        ":error_collector",
        ":types_util",
        "//tensorflow/lite/python/metrics:converter_error_data_proto_cc",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "error_collector_inst_test",
    srcs = ["error_collector_inst_test.cc"],
    data = [
        "testdata/strided_slice.mlir",
    ],
    deps = [
        ":error_collector_inst",
        ":types_util",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:resource_loader",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "types_util",
    srcs = ["types_util.cc"],
    hdrs = ["types_util.h"],
    deps = [
        "//tensorflow/lite/python/metrics:converter_error_data_proto_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
