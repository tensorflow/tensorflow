load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["@llvm-project//mlir:subpackages"],
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
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)
