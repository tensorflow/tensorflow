load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

exports_files(glob([
    "testdata/*.bin",
]))

cc_library(
    name = "quantization_utils",
    srcs = ["quantization_utils.cc"],
    hdrs = ["quantization_utils.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels/internal:round",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

tf_cc_test(
    name = "quantization_utils_test",
    srcs = ["quantization_utils_test.cc"],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantization_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantize_weights",
    srcs = ["quantize_weights.cc"],
    hdrs = ["quantize_weights.h"],
    deps = [
        ":quantization_utils",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
        "//tensorflow/lite:framework",
        # TODO(suharshs): Move the relevant quantization utils to a non-internal location.
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/core:tflite_portable_logging",
    ],
)

cc_library(
    name = "calibrator_lib",
    srcs = ["calibrator.cc"],
    hdrs = ["calibrator.h"],
    deps = [
        ":calibration_common",
        ":calibration_logger",
        ":calibration_reader",
        ":logging_op_resolver",
        ":node_info_delegate",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "calibrator_test",
    srcs = ["calibrator_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/lite:testdata/multi_add.bin)",
    ],
    data = [
        "//tensorflow/lite:testdata/multi_add.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":calibrator_lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "logging_op_resolver",
    srcs = ["logging_op_resolver.cc"],
    hdrs = ["logging_op_resolver.h"],
    deps = [
        ":calibration_common",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/api",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "logging_op_resolver_test",
    srcs = ["logging_op_resolver_test.cc"],
    deps = [
        ":logging_op_resolver",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "calibration_reader",
    srcs = ["calibration_reader.cc"],
    hdrs = ["calibration_reader.h"],
    deps = [
        ":calibration_logger",
        "//tensorflow/lite:framework",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "calibration_logger",
    hdrs = ["calibration_logger.h"],
    deps = [
        "//tensorflow/lite/c:c_api_internal",
    ],
)

cc_library(
    name = "calibration_common",
    hdrs = ["calibration_common.h"],
    deps = [
        "//tensorflow/lite:framework",
    ],
)

cc_library(
    name = "node_info_delegate",
    srcs = ["node_info_delegate.cc"],
    hdrs = ["node_info_delegate.h"],
    deps = [
        ":calibration_common",
        "//tensorflow/lite:framework",
    ],
)

tf_cc_test(
    name = "node_info_delegate_test",
    srcs = ["node_info_delegate_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/lite/tools/optimize:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        "//tensorflow/lite/tools/optimize:testdata/single_conv_weights_min_0_max_plus_10.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":node_info_delegate",
        ":test_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "quantize_weights_test",
    srcs = ["quantize_weights_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/lite/tools/optimize:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        "//tensorflow/lite/tools/optimize:testdata/single_conv_weights_min_0_max_plus_10.bin",
        "//tensorflow/lite/tools/optimize:testdata/weight_shared_between_convs.bin",
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
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "subgraph_quantizer",
    srcs = ["subgraph_quantizer.cc"],
    hdrs = ["subgraph_quantizer.h"],
    deps = [
        ":quantization_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels/internal:round",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/api",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "subgraph_quantizer_test",
    srcs = ["subgraph_quantizer_test.cc"],
    args = [
        "--test_model_file=$(location //tensorflow/lite/tools/optimize:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        "//tensorflow/lite/tools/optimize:testdata/single_avg_pool_min_minus_5_max_plus_5.bin",
        "//tensorflow/lite/tools/optimize:testdata/single_conv_weights_min_0_max_plus_10.bin",
        "//tensorflow/lite/tools/optimize:testdata/single_conv_weights_min_minus_127_max_plus_127.bin",
        "//tensorflow/lite/tools/optimize:testdata/single_softmax_min_minus_5_max_plus_5.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":subgraph_quantizer",
        ":test_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantize_model",
    srcs = ["quantize_model.cc"],
    hdrs = ["quantize_model.h"],
    deps = [
        ":subgraph_quantizer",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
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
        "//tensorflow/lite/tools/optimize:testdata/single_conv_weights_min_0_max_plus_10.bin",
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
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tflite_portable_test_suite()
