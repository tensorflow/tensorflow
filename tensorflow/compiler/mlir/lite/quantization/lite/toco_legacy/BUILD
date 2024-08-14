load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:private",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "portable_tensor_utils",
    srcs = ["portable_tensor_utils.cc"],
    hdrs = ["portable_tensor_utils.h"],
    visibility = [
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib:__pkg__",
    ],
)

cc_library(
    name = "model_utils",
    srcs = ["model_utils.cc"],
    hdrs = ["model_utils.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "model_utils_test",
    srcs = ["model_utils_test.cc"],
    deps = [
        ":model_utils",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "quantization_utils",
    srcs = ["quantization_utils.cc"],
    hdrs = ["quantization_utils.h"],
    deps = [
        ":model_utils",
        ":portable_tensor_utils",
        "//tensorflow/compiler/mlir/lite/kernels/internal:runtime_shape",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
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
    deps = [
        ":quantization_utils",
        "//tensorflow/compiler/mlir/lite/quantization/lite:test_util",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "//tensorflow/lite/core:framework",  # to remove when mlir version is ready.
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:platform_port",
        "@local_xla//xla/tsl/util:command_line_flags",
    ],
)

cc_library(
    name = "quantize_weights",
    srcs = ["quantize_weights.cc"],
    hdrs = ["quantize_weights.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/mlir/lite/quantization/lite:quantize_weights",
        "//tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy:model_utils",
        "//tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy:portable_tensor_utils",
        "//tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy:quantization_utils",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@flatbuffers//:runtime_cc",
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
    deps = [
        ":quantize_weights",
        "//tensorflow/compiler/mlir/lite/quantization/lite:test_util",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "//tensorflow/core:framework_internal",
        "//tensorflow/lite/core:framework",  # to remove when mlir version is ready.
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:platform_port",
        "@local_xla//xla/tsl/util:command_line_flags",
    ],
)
