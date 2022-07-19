# Description: Tests for SavedModel model roundtrip via MLIR TFG module

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
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
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core/ir/importexport:savedmodel_export",
        "//tensorflow/core/ir/importexport:savedmodel_import",
        "//tensorflow/core/ir/importexport/tests/roundtrip",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
    ],
)
