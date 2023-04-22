load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")

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
        "//tensorflow/compiler/mlir/lite/...",
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
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:flatbuffer_translate_lib",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_quantize",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "tfl_to_std",
    srcs = [
        "tfl_to_std.cc",
    ],
    hdrs = [
        "tfl_to_std.h",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_utils.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
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
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
    ],
)

tf_cc_test(
    name = "quantize_model_test",
    srcs = ["quantize_model_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/lite/tools/optimize:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        "//tensorflow/lite/tools/optimize:testdata/add_with_const_input.bin",
        "//tensorflow/lite/tools/optimize:testdata/argmax.bin",
        "//tensorflow/lite/tools/optimize:testdata/broadcast_to.bin",
        "//tensorflow/lite/tools/optimize:testdata/concat.bin",
        "//tensorflow/lite/tools/optimize:testdata/fc.bin",
        "//tensorflow/lite/tools/optimize:testdata/fc_qat.bin",
        "//tensorflow/lite/tools/optimize:testdata/gather_nd.bin",
        "//tensorflow/lite/tools/optimize:testdata/lstm_calibrated.bin",
        "//tensorflow/lite/tools/optimize:testdata/lstm_calibrated2.bin",
        "//tensorflow/lite/tools/optimize:testdata/lstm_quantized.bin",
        "//tensorflow/lite/tools/optimize:testdata/lstm_quantized2.bin",
        "//tensorflow/lite/tools/optimize:testdata/maximum.bin",
        "//tensorflow/lite/tools/optimize:testdata/minimum.bin",
        "//tensorflow/lite/tools/optimize:testdata/mixed.bin",
        "//tensorflow/lite/tools/optimize:testdata/mixed16x8.bin",
        "//tensorflow/lite/tools/optimize:testdata/multi_input_add_reshape.bin",
        "//tensorflow/lite/tools/optimize:testdata/pack.bin",
        "//tensorflow/lite/tools/optimize:testdata/single_avg_pool_min_minus_5_max_plus_5.bin",
        "//tensorflow/lite/tools/optimize:testdata/single_conv_no_bias.bin",
        "//tensorflow/lite/tools/optimize:testdata/single_conv_weights_min_0_max_plus_10.bin",
        "//tensorflow/lite/tools/optimize:testdata/single_conv_weights_min_minus_127_max_plus_127.bin",
        "//tensorflow/lite/tools/optimize:testdata/single_softmax_min_minus_5_max_plus_5.bin",
        "//tensorflow/lite/tools/optimize:testdata/split.bin",
        "//tensorflow/lite/tools/optimize:testdata/svdf_calibrated.bin",
        "//tensorflow/lite/tools/optimize:testdata/svdf_quantized.bin",
        "//tensorflow/lite/tools/optimize:testdata/transpose.bin",
        "//tensorflow/lite/tools/optimize:testdata/unidirectional_sequence_lstm_calibrated.bin",
        "//tensorflow/lite/tools/optimize:testdata/unidirectional_sequence_lstm_quantized.bin",
        "//tensorflow/lite/tools/optimize:testdata/unpack.bin",
        "//tensorflow/lite/tools/optimize:testdata/where.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantize_model",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "//tensorflow/lite/tools/optimize:test_util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)
