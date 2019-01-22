package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/lite/experimental/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

cc_library(
    name = "micro_framework",
    srcs = [
        "debug_log.cc",
        "debug_log_numbers.cc",
        "micro_error_reporter.cc",
        "micro_interpreter.cc",
        "micro_mutable_op_resolver.cc",
        "simple_tensor_allocator.cc",
    ],
    hdrs = [
        "compatibility.h",
        "debug_log.h",
        "debug_log_numbers.h",
        "micro_error_reporter.h",
        "micro_interpreter.h",
        "micro_mutable_op_resolver.h",
        "simple_tensor_allocator.h",
    ],
    deps = [
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

tflite_micro_cc_test(
    name = "micro_error_reporter_test",
    srcs = [
        "micro_error_reporter_test.cc",
    ],
    deps = [
        ":micro_framework",
    ],
)

tflite_micro_cc_test(
    name = "micro_mutable_op_resolver_test",
    srcs = [
        "micro_mutable_op_resolver_test.cc",
    ],
    deps = [
        ":micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "micro_interpreter_test",
    srcs = [
        "micro_interpreter_test.cc",
    ],
    deps = [
        ":micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "simple_tensor_allocator_test",
    srcs = [
        "simple_tensor_allocator_test.cc",
    ],
    deps = [
        ":micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)
