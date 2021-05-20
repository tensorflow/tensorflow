load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "micro_copts",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

package_group(
    name = "micro",
    packages = ["//tensorflow/lite/micro/..."],
)

cc_library(
    name = "micro_compatibility",
    hdrs = [
        "compatibility.h",
    ],
    copts = micro_copts(),
)

cc_library(
    # TODO(b/187093492): Rename to micro_interpreter.
    name = "micro_framework",
    srcs = [
        "micro_interpreter.cc",
    ],
    hdrs = [
        "micro_interpreter.h",
    ],
    copts = micro_copts(),
    deps = [
        ":memory_helpers",
        ":micro_allocator",
        ":micro_error_reporter",
        ":micro_graph",
        ":micro_profiler",
        ":op_resolvers",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "micro_graph",
    srcs = ["micro_graph.cc"],
    hdrs = ["micro_graph.h"],
    deps = [
        ":memory_helpers",
        ":micro_allocator",
        ":micro_error_reporter",
        ":micro_profiler",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "mock_micro_graph",
    srcs = ["mock_micro_graph.cc"],
    hdrs = ["mock_micro_graph.h"],
    deps = [
        ":micro_allocator",
        ":micro_graph",
        ":test_helpers",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "micro_allocator",
    srcs = [
        "micro_allocator.cc",
        "simple_memory_allocator.cc",
    ],
    hdrs = [
        "micro_allocator.h",
        "simple_memory_allocator.h",
    ],
    copts = micro_copts(),
    deps = [
        ":memory_helpers",
        ":micro_compatibility",
        ":micro_error_reporter",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/core/api:op_resolver",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/micro/memory_planner",
        "//tensorflow/lite/micro/memory_planner:greedy_memory_planner",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "memory_helpers",
    srcs = ["memory_helpers.cc"],
    hdrs = ["memory_helpers.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels/internal:reference",
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
    copts = micro_copts(),
    deps = [
        ":micro_utils",
        ":op_resolvers",
        "//tensorflow/lite:type_to_tflitetype",
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
    copts = micro_copts(),
    deps = [
        ":micro_compatibility",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/micro/kernels:conv",
        "//tensorflow/lite/micro/kernels:depthwise_conv",
        "//tensorflow/lite/micro/kernels:ethosu",
        "//tensorflow/lite/micro/kernels:fully_connected",
        "//tensorflow/lite/micro/kernels:micro_ops",
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
    copts = micro_copts(),
)

cc_library(
    name = "micro_time",
    srcs = [
        "micro_time.cc",
    ],
    hdrs = [
        "micro_time.h",
    ],
    copts = micro_copts() + ["-DTF_LITE_USE_CTIME"],
    deps = ["//tensorflow/lite/c:common"],
)

cc_library(
    name = "micro_profiler",
    srcs = [
        "micro_profiler.cc",
    ],
    hdrs = [
        "micro_profiler.h",
    ],
    copts = micro_copts(),
    deps = [
        ":micro_error_reporter",
        ":micro_time",
        "//tensorflow/lite/kernels/internal:compatibility",
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
    copts = micro_copts(),
    deps = [
        ":micro_allocator",
        ":micro_compatibility",
        ":micro_framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_library(
    name = "system_setup",
    srcs = [
        "system_setup.cc",
    ],
    hdrs = [
        "system_setup.h",
    ],
    copts = micro_copts(),
)

cc_test(
    name = "micro_error_reporter_test",
    srcs = [
        "micro_error_reporter_test.cc",
    ],
    deps = [
        ":micro_error_reporter",
    ],
)

cc_test(
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

cc_test(
    name = "micro_interpreter_test",
    srcs = [
        "micro_interpreter_test.cc",
    ],
    deps = [
        ":micro_compatibility",
        ":micro_error_reporter",
        ":micro_framework",
        ":micro_profiler",
        ":micro_utils",
        ":op_resolvers",
        ":recording_allocators",
        ":test_helpers",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
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

cc_test(
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

cc_test(
    name = "micro_allocator_test",
    srcs = [
        "micro_allocator_test.cc",
    ],
    deps = [
        ":memory_helpers",
        ":micro_allocator",
        ":micro_error_reporter",
        ":test_helpers",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/micro/testing:test_conv_model",
    ],
)

cc_test(
    name = "recording_micro_allocator_test",
    srcs = [
        "recording_micro_allocator_test.cc",
    ],
    deps = [
        ":micro_allocator",
        ":micro_error_reporter",
        ":op_resolvers",
        ":recording_allocators",
        ":test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/micro/testing:test_conv_model",
    ],
)

cc_test(
    name = "memory_helpers_test",
    srcs = [
        "memory_helpers_test.cc",
    ],
    deps = [
        ":memory_helpers",
        ":test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "testing_helpers_test",
    srcs = [
        "testing_helpers_test.cc",
    ],
    deps = [
        ":micro_framework",
        "//tensorflow/lite/micro:test_helpers",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "micro_utils_test",
    srcs = [
        "micro_utils_test.cc",
    ],
    deps = [
        ":micro_utils",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "micro_string_test",
    srcs = [
        "micro_string_test.cc",
    ],
    deps = [
        ":micro_string",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "micro_time_test",
    srcs = [
        "micro_time_test.cc",
    ],
    deps = [
        ":micro_time",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "memory_arena_threshold_test",
    srcs = [
        "memory_arena_threshold_test.cc",
    ],
    deps = [
        ":micro_error_reporter",
        ":op_resolvers",
        ":recording_allocators",
        "//tensorflow/lite/micro/benchmarks:keyword_scrambled_model_data",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/micro/testing:test_conv_model",
    ],
)

bzl_library(
    name = "build_def_bzl",
    srcs = ["build_def.bzl"],
    visibility = [":micro"],
)
