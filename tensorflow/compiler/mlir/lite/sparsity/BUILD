load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":friends",
        "//tensorflow:__pkg__",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//learning/brain/experimental/mlir/...",
        "//tensorflow/compiler/mlir/lite/...",
        "//tensorflow/lite/...",
    ],
)

cc_library(
    name = "sparsify_model",
    srcs = [
        "sparsify_model.cc",
    ],
    hdrs = [
        "sparsify_model.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:flatbuffer_translate_lib",
        "//tensorflow/compiler/mlir/lite:pass_registry_utils",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_d2s",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/tools/optimize:reduced_precision_metadata",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "sparsify_model_test",
    srcs = ["sparsify_model_test.cc"],
    data = [
        ":testdata/sparse_tensor.bin",
    ],
    deps = [
        ":sparsify_model",
        "//tensorflow/compiler/mlir/lite/core:absl_error_model_builder",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/tools/optimize:reduced_precision_metadata",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)
