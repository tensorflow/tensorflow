load(
    "//tensorflow/lite/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "cc_library",
    "micro_copts",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "micro_compatibility",
    hdrs = [
        "compatibility.h",
    ],
    build_for_embedded = True,
    copts = micro_copts(),
)

cc_library(
    name = "micro_framework",
    srcs = [
        "memory_helpers.cc",
        "micro_allocator.cc",
        "micro_interpreter.cc",
        "micro_optional_debug_tools.cc",
        "simple_memory_allocator.cc",
    ],
    hdrs = [
        "memory_helpers.h",
        "micro_allocator.h",
        "micro_interpreter.h",
        "micro_optional_debug_tools.h",
        "simple_memory_allocator.h",
    ],
    build_for_embedded = True,
    copts = micro_copts(),
    deps = [
        ":micro_compatibility",
        ":op_resolvers",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/micro/memory_planner",
        "//tensorflow/lite/micro/memory_planner:greedy_memory_planner",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "test_helpers",
    srcs = [
        "test_helpers.cc",
    ],
    hdrs = [
        "test_helpers.h",
    ],
    build_for_embedded = True,
    copts = micro_copts(),
    deps = [
        ":micro_utils",
        ":op_resolvers",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "op_resolvers",
    srcs = [
        "all_ops_resolver.cc",
    ],
    hdrs = [
        "all_ops_resolver.h",
        "micro_mutable_op_resolver.h",
        "micro_op_resolver.h",
    ],
    build_for_embedded = True,
    copts = micro_copts(),
    deps = [
        ":micro_compatibility",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/micro/kernels:micro_ops",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

# TODO(b/144176795): This target should really be handled differently so that we
# do not have a fork in the build graph. The bug has some initial ideas.
cc_library(
    name = "portable_optimized_op_resolver",
    srcs = [
        "all_ops_resolver.cc",
        "micro_mutable_op_resolver.h",
        "micro_op_resolver.h",
    ],
    hdrs = [
        "all_ops_resolver.h",
    ],
    copts = micro_copts(),
    deps = [
        ":micro_compatibility",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/micro/kernels:portable_optimized_micro_ops",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "debug_log",
    srcs = [
        "debug_log.cc",
    ],
    hdrs = [
        "debug_log.h",
    ],
    build_for_embedded = True,
    copts = micro_copts(),
)

cc_library(
    name = "micro_error_reporter",
    srcs = [
        "micro_error_reporter.cc",
    ],
    hdrs = [
        "micro_error_reporter.h",
    ],
    build_for_embedded = True,
    copts = micro_copts(),
    deps = [
        ":debug_log",
        ":micro_compatibility",
        ":micro_string",
        "//tensorflow/lite/core/api",
    ],
)

cc_library(
    name = "micro_string",
    srcs = [
        "micro_string.cc",
    ],
    hdrs = [
        "micro_string.h",
    ],
    build_for_embedded = True,
    copts = micro_copts(),
    deps = ["//tensorflow/lite/c:common"],
)

cc_library(
    name = "micro_time",
    srcs = [
        "posix/micro_time.cc",
    ],
    hdrs = [
        "micro_time.h",
    ],
    build_for_embedded = True,
    copts = micro_copts(),
    deps = ["//tensorflow/lite/c:common"],
)

cc_library(
    name = "micro_utils",
    srcs = [
        "micro_utils.cc",
    ],
    hdrs = [
        "micro_utils.h",
    ],
    build_for_embedded = True,
    copts = micro_copts(),
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:op_macros",
    ],
)

cc_library(
    name = "recording_allocators",
    srcs = [
        "recording_micro_allocator.cc",
        "recording_simple_memory_allocator.cc",
    ],
    hdrs = [
        "recording_micro_allocator.h",
        "recording_micro_interpreter.h",
        "recording_simple_memory_allocator.h",
    ],
    build_for_embedded = True,
    copts = micro_copts(),
    deps = [
        ":micro_compatibility",
        ":micro_framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

tflite_micro_cc_test(
    name = "micro_error_reporter_test",
    srcs = [
        "micro_error_reporter_test.cc",
    ],
    deps = [
        ":micro_error_reporter",
    ],
)

tflite_micro_cc_test(
    name = "micro_mutable_op_resolver_test",
    srcs = [
        "micro_mutable_op_resolver_test.cc",
    ],
    deps = [
        ":micro_framework",
        ":op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "micro_interpreter_test",
    srcs = [
        "micro_interpreter_test.cc",
    ],
    deps = [
        ":micro_framework",
        ":micro_utils",
        ":op_resolvers",
        ":test_helpers",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "simple_memory_allocator_test",
    srcs = [
        "simple_memory_allocator_test.cc",
    ],
    deps = [
        ":micro_framework",
        ":test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "recording_simple_memory_allocator_test",
    srcs = [
        "recording_simple_memory_allocator_test.cc",
    ],
    deps = [
        ":micro_framework",
        ":recording_allocators",
        ":test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "micro_allocator_test",
    srcs = [
        "micro_allocator_test.cc",
    ],
    deps = [
        ":micro_framework",
        ":test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "recording_micro_allocator_test",
    srcs = [
        "recording_micro_allocator_test.cc",
    ],
    deps = [
        ":micro_framework",
        ":op_resolvers",
        ":recording_allocators",
        ":test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/micro/testing:test_conv_model",
    ],
)

tflite_micro_cc_test(
    name = "memory_helpers_test",
    srcs = [
        "memory_helpers_test.cc",
    ],
    deps = [
        ":micro_framework",
        ":test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "testing_helpers_test",
    srcs = [
        "testing_helpers_test.cc",
    ],
    deps = [
        ":micro_framework",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "micro_utils_test",
    srcs = [
        "micro_utils_test.cc",
    ],
    deps = [
        ":micro_utils",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "micro_string_test",
    srcs = [
        "micro_string_test.cc",
    ],
    deps = [
        ":micro_string",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "micro_time_test",
    srcs = [
        "micro_time_test.cc",
    ],
    deps = [
        ":micro_time",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflite_micro_cc_test(
    name = "memory_arena_threshold_test",
    srcs = [
        "memory_arena_threshold_test.cc",
    ],
    deps = [
        ":op_resolvers",
        ":recording_allocators",
        "//tensorflow/lite/micro/benchmarks:keyword_scrambled_model_data",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/micro/testing:test_conv_model",
    ],
)
