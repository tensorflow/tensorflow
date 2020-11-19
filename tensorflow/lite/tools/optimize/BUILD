load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(glob([
    "testdata/*.bin",
]))

cc_library(
    name = "modify_model_interface",
    srcs = ["modify_model_interface.cc"],
    hdrs = ["modify_model_interface.h"],
    deps = [
        ":model_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "modify_model_interface_test",
    srcs = ["modify_model_interface_test.cc"],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":modify_model_interface",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

cc_binary(
    name = "modify_model_interface_main",
    srcs = ["modify_model_interface_main.cc"],
    deps = [":modify_model_interface"],
)

cc_library(
    name = "quantization_wrapper_utils",
    srcs = ["quantization_wrapper_utils.cc"],
    hdrs = ["quantization_wrapper_utils.h"],
    deps = [
        ":operator_property",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

tf_cc_test(
    name = "quantization_wrapper_utils_test",
    srcs = ["quantization_wrapper_utils_test.cc"],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantization_wrapper_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "quantization_wrapper_utils_custom_test",
    srcs = [
        "quantization_wrapper_utils.cc",
        "quantization_wrapper_utils.h",
        "quantization_wrapper_utils_custom_test.cc",
    ],
    defines = [
        "TFLITE_CUSTOM_LSTM",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":operator_property",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "quantization_wrapper",
    srcs = ["quantization_wrapper.cc"],
    hdrs = ["quantization_wrapper.h"],
    deps = [
        ":quantization_wrapper_utils",
        ":quantize_model",
    ],
)

cc_library(
    name = "quantization_utils",
    srcs = ["quantization_utils.cc"],
    hdrs = ["quantization_utils.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":model_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels/internal:cppmath",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/schema:schema_fbs",
        "//third_party/eigen3",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "model_utils",
    srcs = ["model_utils.cc"],
    hdrs = ["model_utils.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":operator_property",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "model_utils_test",
    srcs = ["model_utils_test.cc"],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":model_utils",
        ":test_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "operator_property",
    srcs = ["operator_property.cc"],
    hdrs = ["operator_property.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
    ],
)

tf_cc_test(
    name = "quantization_utils_test",
    srcs = ["quantization_utils_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/lite/tools/optimize:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        ":testdata/single_conv_weights_min_0_max_plus_10.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantization_utils",
        ":test_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "//tensorflow/lite/testing:util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "quantize_weights",
    srcs = ["quantize_weights.cc"],
    hdrs = ["quantize_weights.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":quantization_utils",
        ":model_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/container:flat_hash_map",
        "@flatbuffers",
        "//tensorflow/lite:framework",
        # TODO(suharshs): Move the relevant quantization utils to a non-internal location.
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "//tensorflow/core:tflite_portable_logging",
    ],
)

tf_cc_test(
    name = "quantize_weights_test",
    srcs = ["quantize_weights_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/lite/tools/optimize:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        ":testdata/custom_op.bin",
        ":testdata/quantized_with_gather.bin",
        ":testdata/single_conv_weights_min_0_max_plus_10.bin",
        ":testdata/weight_shared_between_convs.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantize_weights",
        ":test_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//tensorflow/lite/core/api",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "quantize_model",
    srcs = ["quantize_model.cc"],
    hdrs = ["quantize_model.h"],
    deps = [
        ":model_utils",
        ":operator_property",
        ":quantization_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "quantize_model_test",
    srcs = ["quantize_model_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/lite/tools/optimize:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        ":testdata/add_with_const_input.bin",
        ":testdata/argmax.bin",
        ":testdata/concat.bin",
        ":testdata/fc.bin",
        ":testdata/lstm_calibrated.bin",
        ":testdata/lstm_calibrated2.bin",
        ":testdata/lstm_quantized.bin",
        ":testdata/lstm_quantized2.bin",
        ":testdata/maximum.bin",
        ":testdata/minimum.bin",
        ":testdata/mixed.bin",
        ":testdata/mixed16x8.bin",
        ":testdata/multi_input_add_reshape.bin",
        ":testdata/pack.bin",
        ":testdata/single_avg_pool_min_minus_5_max_plus_5.bin",
        ":testdata/single_conv_no_bias.bin",
        ":testdata/single_conv_weights_min_0_max_plus_10.bin",
        ":testdata/single_conv_weights_min_minus_127_max_plus_127.bin",
        ":testdata/single_softmax_min_minus_5_max_plus_5.bin",
        ":testdata/split.bin",
        ":testdata/svdf_calibrated.bin",
        ":testdata/svdf_quantized.bin",
        ":testdata/transpose.bin",
        ":testdata/unidirectional_sequence_lstm_calibrated.bin",
        ":testdata/unidirectional_sequence_lstm_quantized.bin",
        ":testdata/unpack.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantize_model",
        ":test_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tflite_portable_test_suite()
