# Description:
#   XLA evaluator implementation.

load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

cc_library(
    name = "hlo_evaluator",
    srcs = [
        "hlo_evaluator.cc",
        "hlo_evaluator_typed_visitor.h",
        "hlo_evaluator_typed_visitor_bfloat16.cc",
        "hlo_evaluator_typed_visitor_bool.cc",
        "hlo_evaluator_typed_visitor_complex128.cc",
        "hlo_evaluator_typed_visitor_complex64.cc",
        "hlo_evaluator_typed_visitor_double.cc",
        "hlo_evaluator_typed_visitor_float.cc",
        "hlo_evaluator_typed_visitor_float8.cc",
        "hlo_evaluator_typed_visitor_half.cc",
        "hlo_evaluator_typed_visitor_int16.cc",
        "hlo_evaluator_typed_visitor_int32.cc",
        "hlo_evaluator_typed_visitor_int64.cc",
        "hlo_evaluator_typed_visitor_int8.cc",
        "hlo_evaluator_typed_visitor_uint16.cc",
        "hlo_evaluator_typed_visitor_uint32.cc",
        "hlo_evaluator_typed_visitor_uint64.cc",
        "hlo_evaluator_typed_visitor_uint8.cc",
    ],
    hdrs = ["hlo_evaluator.h"],
    deps = [
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:call_graph",
        "//tensorflow/compiler/xla/service:compilation_environments",
        "//tensorflow/compiler/xla/service:dynamic_dimension_inference",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/service:tuple_points_to_analysis",
        "//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_matmul",
        "//tensorflow/tsl/lib/core:bitmap",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "hlo_evaluator_test",
    srcs = ["hlo_evaluator_test.cc"],
    deps = [
        ":hlo_evaluator",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_element_type_converter",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_benchmark",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
)
