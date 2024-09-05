load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = [
        ":friends",
        "//tensorflow:__pkg__",
    ],
    licenses = ["notice"],
)

exports_files(glob([
    "testdata/*.bin",
]))

package_group(
    name = "friends",
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
        "//tensorflow/compiler/mlir/lite:tf_tfl_passes",
        "//tensorflow/compiler/mlir/lite/debug",
        "//tensorflow/compiler/mlir/lite/debug:debug_options_proto_cc",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib:quantization_config",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "quantize_weights",
    srcs = [
        "quantize_weights.cc",
    ],
    hdrs = [
        "quantize_weights.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:flatbuffer_translate_lib",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tf_tfl_passes",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib:quantization_config",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tfl_to_std",
    srcs = [
        "tfl_to_std.cc",
    ],
    hdrs = [
        "tfl_to_std.h",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib:quantization_utils.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib:quantization_config",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
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
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
    ],
)

tf_cc_test(
    name = "quantize_model_test",
    srcs = ["quantize_model_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/add_with_const_input.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/argmax.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/broadcast_to.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/concat.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/fc.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/fc_qat.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/gather_nd.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/lstm_calibrated.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/lstm_calibrated2.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/lstm_quantized.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/lstm_quantized2.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/maximum.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/minimum.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/mixed.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/mixed16x8.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/multi_input_add_reshape.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/pack.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_avg_pool_min_minus_5_max_plus_5.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_conv_no_bias.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_minus_127_max_plus_127.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_softmax_min_minus_5_max_plus_5.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/split.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/svdf_calibrated.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/svdf_quantized.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/transpose.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/unidirectional_sequence_lstm_calibrated.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/unidirectional_sequence_lstm_quantized.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/unpack.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/where.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantize_model",
        ":test_util",
        "//tensorflow/compiler/mlir/lite/core:absl_error_model_builder",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

tf_cc_test(
    name = "quantize_weights_test",
    srcs = ["quantize_weights_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/custom_op.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/quantized_with_gather.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin",
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/weight_shared_between_convs.bin",
    ],
    tags = [
        # TODO(b/327796566): re-enable after the bug is fixed
        "manual",
        "noasan",
        "noguitar",
        "notap",
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantize_weights",
        ":test_util",
        "//tensorflow/compiler/mlir/lite/core:absl_error_model_builder",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/core/api:error_reporter",
        "@com_google_googletest//:gtest",
    ],
)
