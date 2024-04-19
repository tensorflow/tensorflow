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
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:flatbuffer_translate_lib",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_d2s",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:private_c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools/optimize:reduced_precision_support",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "sparsify_model_test",
    srcs = ["sparsify_model_test.cc"],
    data = [
        "//tensorflow/lite:testdata/sparse_tensor.bin",
    ],
    deps = [
        ":sparsify_model",
        "//tensorflow/lite/core:model_builder",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/core/c:private_c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools/optimize:reduced_precision_support",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)
