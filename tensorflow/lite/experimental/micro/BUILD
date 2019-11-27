load(
    "//tensorflow/lite/experimental/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "micro_framework",
    srcs = [
        "debug_log.cc",
        "debug_log_numbers.cc",
        "memory_helpers.cc",
        "micro_allocator.cc",
        "micro_error_reporter.cc",
        "micro_interpreter.cc",
        "micro_mutable_op_resolver.cc",
        "micro_optional_debug_tools.cc",
        "simple_memory_allocator.cc",
        "test_helpers.cc",
    ],
    hdrs = [
        "compatibility.h",
        "debug_log.h",
        "debug_log_numbers.h",
        "memory_helpers.h",
        "micro_allocator.h",
        "micro_error_reporter.h",
        "micro_interpreter.h",
        "micro_mutable_op_resolver.h",
        "micro_optional_debug_tools.h",
        "simple_memory_allocator.h",
        "test_helpers.h",
    ],
    copts = [
        "-Werror",
        "-Wdouble-promotion",
        "-Wsign-compare",
    ],
    deps = [
        ":micro_utils",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/experimental/micro/memory_planner:greedy_memory_planner",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "micro_utils",
    srcs = [
        "micro_utils.cc",
    ],
    hdrs = [
        "micro_utils.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
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
    name = "simple_memory_allocator_test",
    srcs = [
        "simple_memory_allocator_test.cc",
    ],
    deps = [
        ":micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "micro_allocator_test",
    srcs = [
        "micro_allocator_test.cc",
    ],
    deps = [
        ":micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "memory_helpers_test",
    srcs = [
        "memory_helpers_test.cc",
    ],
    deps = [
        ":micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "testing_helpers_test",
    srcs = [
        "testing_helpers_test.cc",
    ],
    deps = [
        ":micro_framework",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "micro_utils_test",
    srcs = [
        "micro_utils_test.cc",
    ],
    deps = [
        ":micro_utils",
        "//tensorflow/lite/experimental/micro/testing:micro_test",
    ],
)
