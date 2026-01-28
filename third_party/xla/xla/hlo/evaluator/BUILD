# Description:
#   XLA evaluator implementation.

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "hlo_evaluator_interface",
    hdrs = ["hlo_evaluator_interface.h"],
    deps = [
        "//xla:literal",
        "//xla/hlo/ir:hlo",
        "//xla/service:dynamic_dimension_inference",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
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
        "hlo_evaluator_typed_visitor_int1.cc",
        "hlo_evaluator_typed_visitor_int16.cc",
        "hlo_evaluator_typed_visitor_int2.cc",
        "hlo_evaluator_typed_visitor_int32.cc",
        "hlo_evaluator_typed_visitor_int4.cc",
        "hlo_evaluator_typed_visitor_int64.cc",
        "hlo_evaluator_typed_visitor_int8.cc",
        "hlo_evaluator_typed_visitor_mxfloat.cc",
        "hlo_evaluator_typed_visitor_uint16.cc",
        "hlo_evaluator_typed_visitor_uint32.cc",
        "hlo_evaluator_typed_visitor_uint64.cc",
        "hlo_evaluator_typed_visitor_uint8.cc",
    ],
    hdrs = ["hlo_evaluator.h"],
    tags = if_google([
        # This header is only conditionally included in hlo_evaluator.cc when a flag is set, which
        # build_cleaner doesn't understand, so we need to tell it to ignore it.
        "ignore_for_dep=third_party/tensorflow/compiler/xla/tsl/framework/contraction/eigen_contraction_kernel.h",
    ]),
    deps = [
        ":hlo_evaluator_interface",
        "//xla:array2d",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:tuple_points_to_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/service:dynamic_dimension_inference",
        "//xla/service:gather_scatter_utils",
        "//xla/service:hlo_module_config",
        "//xla/service:logical_buffer",
        "//xla/service:pattern_matcher",
        "//xla/service:shape_inference",
        "//xla/tsl/lib/core:bitmap",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:ml_dtypes",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "hlo_evaluator_slow_reduce_window_test",
    timeout = "long",
    srcs = ["hlo_evaluator_slow_reduce_window_test.cc"],
    tags = [
        # Times out in asan/tsan/zapfhahn.
        "noasan",
        "notsan",
        "nozapfhahn",
    ],
    deps = [
        ":hlo_evaluator",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "hlo_evaluator_test",
    srcs = ["hlo_evaluator_test.cc"],
    deps = [
        ":hlo_evaluator",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:comparison_util",
        "//xla:debug_options_flags",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:tuple_points_to_analysis",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/transforms/simplifiers:hlo_element_type_converter",
        "//xla/service:dynamic_dimension_inference",
        "//xla/service:hlo_module_config",
        "//xla/service:shape_inference",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_utils",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",  # fixdeps: keep
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "caching_hlo_evaluator",
    srcs = ["caching_hlo_evaluator.cc"],
    hdrs = ["caching_hlo_evaluator.h"],
    deps = [
        ":hlo_evaluator_interface",
        "//xla:literal",
        "//xla/hlo/ir:hlo",
        "//xla/service:dynamic_dimension_inference",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:fingerprint",
        "@tsl//tsl/platform:path",
    ],
)

xla_cc_test(
    name = "caching_hlo_evaluator_test",
    srcs = ["caching_hlo_evaluator_test.cc"],
    deps = [
        ":caching_hlo_evaluator",
        ":hlo_evaluator_interface",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/service:dynamic_dimension_inference",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",  # fixdeps: keep
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:path",
    ],
)
