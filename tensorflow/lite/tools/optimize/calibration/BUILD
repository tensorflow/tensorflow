load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "builtin_logging_op",
    srcs = ["builtin_logging_ops/lstm.cc"],
    hdrs = ["builtin_logging_ops/lstm.h"],
    copts = tflite_copts(),
    deps = [
        ":calibration_logger",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:headers",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:lstm_shared",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels/internal:kernel_utils",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:reference",
        "//tensorflow/lite/kernels/internal:tensor_utils",
    ],
)

cc_library(
    name = "custom_logging_op",
    srcs = ["custom_logging_ops/lstm.cc"],
    hdrs = ["custom_logging_ops/lstm.h"],
    copts = tflite_copts(),
    deps = [
        ":calibration_logger",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:headers",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:lstm_shared",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels/internal:kernel_utils",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:reference",
        "//tensorflow/lite/kernels/internal:tensor_utils",
    ],
)

cc_library(
    name = "calibrator_lib",
    srcs = ["calibrator.cc"],
    hdrs = ["calibrator.h"],
    copts = tflite_copts(),
    deps = [
        ":builtin_logging_op",
        ":calibration_common",
        ":calibration_logger",
        ":calibration_reader",
        ":custom_logging_op",
        ":logging_op",
        ":logging_op_resolver",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/container:flat_hash_map",
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
        "//tensorflow/lite:testdata/call_once_mul.bin",
        "//tensorflow/lite:testdata/custom_lstm.bin",
        "//tensorflow/lite:testdata/lstm.bin",
        "//tensorflow/lite:testdata/multi_add.bin",
        "//tensorflow/lite:testdata/multi_subgraphs_while.bin",
        "//tensorflow/lite:testdata/unidirectional_sequence_lstm.bin",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "logging_op_resolver",
    srcs = ["logging_op_resolver.cc"],
    hdrs = ["logging_op_resolver.h"],
    copts = tflite_copts(),
    deps = [
        ":calibration_common",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/api",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "logging_op_resolver_test",
    srcs = ["logging_op_resolver_test.cc"],
    deps = [
        ":logging_op_resolver",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "calibration_reader",
    srcs = ["calibration_reader.cc"],
    hdrs = ["calibration_reader.h"],
    copts = tflite_copts(),
    deps = [
        ":calibration_logger",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "calibration_logger",
    srcs = ["calibration_logger.cc"],
    hdrs = ["calibration_logger.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "calibration_common",
    hdrs = ["calibration_common.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite:framework",
    ],
)

cc_library(
    name = "logging_op",
    hdrs = ["logging_op.h"],
    copts = tflite_copts(),
    deps = [
        ":calibration_logger",
        "//tensorflow/lite/core/c:common",
    ],
)

tflite_portable_test_suite()
