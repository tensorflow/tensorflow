# Description: Tests for TFG graph transform tool

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = ["//visibility:__subpackages__"],
    licenses = ["notice"],
)

filegroup(
    name = "saved_model_test_files",
    srcs = [
        "savedmodel_v1/saved_model.pb",
        "savedmodel_v2/saved_model.pb",
    ],
)

tf_cc_test(
    name = "saved_model_roundtrip_test",
    srcs = ["saved_model_roundtrip_test.cc"],
    data = [
        ":saved_model_test_files",
    ],
    deps = [
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/importexport/tests/roundtrip",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/tools/tfg_graph_transforms:export",
        "//tensorflow/tools/tfg_graph_transforms:import",
        "//tensorflow/tools/tfg_graph_transforms:utils",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
    ],
)
