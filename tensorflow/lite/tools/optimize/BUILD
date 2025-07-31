load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "reduced_precision_support",
    srcs = [],
    hdrs = [
        "reduced_precision_support.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/lite/tools/optimize:reduced_precision_metadata",
    ],
)

tf_cc_test(
    name = "reduced_precision_support_test",
    srcs = ["reduced_precision_support_test.cc"],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":reduced_precision_support",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "//tensorflow/lite/testing:util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "modify_model_interface",
    srcs = ["modify_model_interface.cc"],
    hdrs = ["modify_model_interface.h"],
    deps = [
        ":model_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/container:flat_hash_map",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_binary(
    name = "modify_model_interface_main",
    srcs = ["modify_model_interface_main.cc"],
    deps = [
        ":modify_model_interface",
        ":quantize_model",
    ],
)

cc_library(
    name = "quantization_wrapper_utils",
    srcs = ["quantization_wrapper_utils.cc"],
    hdrs = ["quantization_wrapper_utils.h"],
    deps = [
        ":operator_property",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantization_wrapper",
    srcs = ["quantization_wrapper.cc"],
    hdrs = ["quantization_wrapper.h"],
    deps = [
        ":quantization_wrapper_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools/optimize:quantize_model",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantization_utils",
    srcs = ["quantization_utils.cc"],
    hdrs = ["quantization_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":model_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels/internal:cppmath",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/schema:schema_fbs",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "model_utils",
    srcs = ["model_utils.cc"],
    hdrs = ["model_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":operator_property",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "model_utils_test",
    srcs = ["model_utils_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":model_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

alias(
    name = "operator_property",
    actual = "//tensorflow/compiler/mlir/lite/tools/optimize:operator_property",
)

tf_cc_test(
    name = "quantization_utils_test",
    srcs = ["quantization_utils_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantization_utils",
        "//tensorflow/compiler/mlir/lite/quantization/lite:test_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "//tensorflow/lite/testing:util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
        "@flatbuffers",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels/internal:cppmath",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
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
        "//tensorflow/compiler/mlir/lite/quantization/lite:testdata/resource_vars_calibrated.bin",
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
        "//tensorflow/compiler/mlir/lite/quantization/lite:test_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tflite_portable_test_suite()
