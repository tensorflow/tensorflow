load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(licenses = ["notice"])

tf_cc_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.cc"],
    data = [
        "testdata/test.mlir",
    ],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt:import_model",
        "//tensorflow/compiler/mlir/tfrt:saved_model",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:resource_loader",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)
