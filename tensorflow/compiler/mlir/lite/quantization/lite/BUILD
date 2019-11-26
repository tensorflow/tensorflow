load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["@local_config_mlir//:subpackages"],
    packages = [
        "//learning/brain/experimental/mlir/...",
        "//tensorflow/lite/...",
    ],
)

cc_library(
    name = "quantize_model",
    srcs = [
        "quantize_model.cc",
    ],
    hdrs = [
        "quantize_model.h",
        "//tensorflow/compiler/mlir/lite:transforms/passes.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:flatbuffer_translate_lib",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_quantize",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
    ],
)

# Binary to apply quantization on the annotated files.
tf_cc_binary(
    name = "tfl_quantizer",
    srcs = [
        "tfl_quantizer.cc",
    ],
    deps = [
        ":quantize_model",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
    ],
)
