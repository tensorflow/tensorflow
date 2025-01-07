# Description:
#   Implementation of XLA’s HLO transformations.

load("//xla:xla.bzl", "xla_cc_test")
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
    name = "bfloat16_propagation",
    srcs = ["bfloat16_propagation.cc"],
    hdrs = ["bfloat16_propagation.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "//xla/service:float_support",
        "//xla/service:hlo_value",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "bfloat16_propagation_test",
    srcs = ["bfloat16_propagation_test.cc"],
    deps = [
        ":bfloat16_propagation",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:float_support",
        "//xla/service:hlo_verifier",
        "//xla/tests:literal_test_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "op_expander_pass",
    srcs = ["expanders/op_expander_pass.cc"],
    hdrs = ["expanders/op_expander_pass.h"],
    deps = [
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "optimization_barrier_expander",
    srcs = ["expanders/optimization_barrier_expander.cc"],
    hdrs = ["expanders/optimization_barrier_expander.h"],
    deps = [
        ":op_expander_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "comparison_expander",
    srcs = ["expanders/comparison_expander.cc"],
    hdrs = ["expanders/comparison_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "cholesky_expander",
    srcs = ["expanders/cholesky_expander.cc"],
    hdrs = ["expanders/cholesky_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:constants",
        "//xla/hlo/builder/lib:loops",
        "//xla/hlo/builder/lib:math",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/builder/lib:slicing",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "qr_expander",
    srcs = ["expanders/qr_expander.cc"],
    hdrs = ["expanders/qr_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:constants",
        "//xla/hlo/builder/lib:loops",
        "//xla/hlo/builder/lib:math",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/builder/lib:qr",
        "//xla/hlo/builder/lib:slicing",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "real_imag_expander",
    srcs = ["expanders/real_imag_expander.cc"],
    hdrs = ["expanders/real_imag_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "real_imag_expander_test",
    size = "small",
    srcs = ["expanders/real_imag_expander_test.cc"],
    deps = [
        ":real_imag_expander",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_creation_utils",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "eigh_expander",
    srcs = ["expanders/eigh_expander.cc"],
    hdrs = ["expanders/eigh_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:comparators",
        "//xla/hlo/builder/lib:constants",
        "//xla/hlo/builder/lib:loops",
        "//xla/hlo/builder/lib:math",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/builder/lib:slicing",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "convolution_4d_expander",
    srcs = ["expanders/convolution_4d_expander.cc"],
    hdrs = ["expanders/convolution_4d_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "convolution_4d_expander_test",
    srcs = ["expanders/convolution_4d_expander_test.cc"],
    deps = [
        "convolution_4d_expander",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "convolution_pred_expander",
    srcs = ["expanders/convolution_pred_expander.cc"],
    hdrs = ["expanders/convolution_pred_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:pattern_matcher",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "convolution_pred_expander_test",
    srcs = ["expanders/convolution_pred_expander_test.cc"],
    deps = [
        ":convolution_pred_expander",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "logistic_expander",
    srcs = ["expanders/logistic_expander.cc"],
    hdrs = ["expanders/logistic_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "logistic_expander_test",
    srcs = ["expanders/logistic_expander_test.cc"],
    deps = [
        ":logistic_expander",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:dynamic_padder",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "bitcast_dtypes_expander",
    srcs = ["expanders/bitcast_dtypes_expander.cc"],
    hdrs = ["expanders/bitcast_dtypes_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:broadcast",
        "//xla/hlo/builder/lib:constants",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "bitcast_dtypes_expander_test",
    srcs = ["expanders/bitcast_dtypes_expander_test.cc"],
    deps = [
        ":bitcast_dtypes_expander",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "while_loop_trip_count_annotator",
    srcs = ["while_loop_trip_count_annotator.cc"],
    hdrs = ["while_loop_trip_count_annotator.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:while_loop_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "while_loop_trip_count_annotator_test",
    srcs = ["while_loop_trip_count_annotator_test.cc"],
    deps = [
        ":while_loop_trip_count_annotator",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "defuser",
    srcs = ["defuser.cc"],
    hdrs = ["defuser.h"],
    deps = [
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:call_graph",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "defuser_test",
    srcs = ["defuser_test.cc"],
    deps = [
        ":defuser",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "despecializer_test",
    srcs = ["despecializer_test.cc"],
    deps = [
        ":despecializer",
        "//xla:literal",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "dot_decomposer",
    srcs = ["expanders/dot_decomposer.cc"],
    hdrs = ["expanders/dot_decomposer.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:shape_inference",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "dot_decomposer_test",
    srcs = ["expanders/dot_decomposer_test.cc"],
    deps = [
        ":dot_decomposer",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",  # fixdeps: keep
    ],
)

cc_library(
    name = "reshape_decomposer",
    srcs = ["expanders/reshape_decomposer.cc"],
    hdrs = ["expanders/reshape_decomposer.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "reduce_decomposer",
    srcs = ["expanders/reduce_decomposer.cc"],
    hdrs = ["expanders/reduce_decomposer.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "reduce_decomposer_test",
    srcs = ["expanders/reduce_decomposer_test.cc"],
    deps = [
        ":reduce_decomposer",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "reshape_decomposer_test",
    srcs = ["expanders/reshape_decomposer_test.cc"],
    deps = [
        ":reshape_decomposer",
        "//xla:test",
        "//xla:test_helpers",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "memory_space_propagation",
    srcs = ["memory_space_propagation.cc"],
    hdrs = ["memory_space_propagation.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/analysis:hlo_dataflow_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "memory_space_propagation_test",
    srcs = ["memory_space_propagation_test.cc"],
    deps = [
        ":memory_space_propagation",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "stable_sort_expander",
    srcs = ["expanders/stable_sort_expander.cc"],
    hdrs = ["expanders/stable_sort_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "stable_sort_expander_test",
    srcs = ["expanders/stable_sort_expander_test.cc"],
    deps = [
        ":stable_sort_expander",
        "//xla:test",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/transforms/simplifiers:algebraic_simplifier",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "convert_memory_placement_to_internal_annotations",
    srcs = ["convert_memory_placement_to_internal_annotations.cc"],
    hdrs = ["convert_memory_placement_to_internal_annotations.h"],
    deps = [
        "//xla:side_effect_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:host_memory_offload_annotations_hdr",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "convert_memory_placement_to_internal_annotations_test",
    srcs = ["convert_memory_placement_to_internal_annotations_test.cc"],
    deps = [
        ":convert_memory_placement_to_internal_annotations",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:host_memory_offload_annotations_hdr",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "host_offload_legalize",
    srcs = ["host_offload_legalize.cc"],
    hdrs = ["host_offload_legalize.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:call_graph",
        "//xla/service:hlo_value",
        "//xla/service:host_memory_offload_annotations_hdr",
        "//xla/service:host_offload_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "host_offload_legalize_test",
    srcs = ["host_offload_legalize_test.cc"],
    shard_count = 12,
    deps = [
        ":host_offload_legalize",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:host_memory_offload_annotations_hdr",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "host_offloader",
    srcs = ["host_offloader.cc"],
    hdrs = ["host_offloader.h"],
    deps = [
        "//xla:literal_util",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:call_graph",
        "//xla/service:hlo_buffer",
        "//xla/service:hlo_cse",
        "//xla/service:hlo_value",
        "//xla/service:host_memory_offload_annotations_hdr",
        "//xla/service:host_offload_utils",
        "//xla/service:pattern_matcher",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "host_offloader_test",
    srcs = ["host_offloader_test.cc"],
    shard_count = 12,
    deps = [
        ":host_offload_legalize",
        ":host_offloader",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:hlo_verifier",
        "//xla/service:host_memory_offload_annotations_hdr",
        "//xla/service:host_offload_utils",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "host_offloading_prepare",
    srcs = ["host_offloading_prepare.cc"],
    hdrs = ["host_offloading_prepare.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:call_graph",
        "//xla/service:host_memory_offload_annotations_hdr",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "host_offloading_prepare_test",
    srcs = ["host_offloading_prepare_test.cc"],
    deps = [
        ":host_offloading_prepare",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:host_memory_offload_annotations_hdr",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "despecializer",
    srcs = ["despecializer.cc"],
    hdrs = ["despecializer.h"],
    deps = [
        ":defuser",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms/simplifiers:float_normalization",
        "//xla/hlo/transforms/simplifiers:hlo_memory_scheduler",
        "//xla/hlo/transforms/simplifiers:sub_byte_normalization",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "literal_canonicalizer",
    srcs = ["literal_canonicalizer.cc"],
    hdrs = ["literal_canonicalizer.h"],
    deps = [
        "//xla:literal_pool",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "literal_canonicalizer_test",
    srcs = ["literal_canonicalizer_test.cc"],
    deps = [
        ":literal_canonicalizer",
        "//xla:literal_pool",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "dynamic_index_splitter",
    srcs = ["expanders/dynamic_index_splitter.cc"],
    hdrs = ["expanders/dynamic_index_splitter.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "dynamic_index_splitter_test",
    srcs = ["expanders/dynamic_index_splitter_test.cc"],
    deps = [
        ":dynamic_index_splitter",
        "//xla:test",
        "//xla:test_helpers",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "rng_expander",
    srcs = ["expanders/rng_expander.cc"],
    hdrs = ["expanders/rng_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:prng",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "rng_bit_generator_expander",
    srcs = ["expanders/rng_bit_generator_expander.cc"],
    hdrs = ["expanders/rng_bit_generator_expander.h"],
    deps = [
        ":op_expander_pass",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder/lib:prng",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "operand_upcaster",
    srcs = ["operand_upcaster.cc"],
    hdrs = ["operand_upcaster.h"],
    deps = [
        ":op_expander_pass",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_creation_utils",
        "//xla/service:shape_inference",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "operand_upcaster_test",
    srcs = ["operand_upcaster_test.cc"],
    deps = [
        ":operand_upcaster",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "stochastic_convert_decomposer",
    srcs = ["expanders/stochastic_convert_decomposer.cc"],
    hdrs = ["expanders/stochastic_convert_decomposer.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_creation_utils",
        "//xla/service:shape_inference",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "stochastic_convert_decomposer_test",
    srcs = ["expanders/stochastic_convert_decomposer_test.cc"],
    deps = [
        ":stochastic_convert_decomposer",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "sharding_format_picker",
    testonly = True,
    srcs = ["sharding_format_picker.cc"],
    hdrs = ["sharding_format_picker.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:tile_assignment",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "add_original_value",
    srcs = ["add_original_value.cc"],
    hdrs = ["add_original_value.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "add_original_value_test",
    srcs = ["add_original_value_test.cc"],
    deps = [
        ":add_original_value",
        "//xla:shape_util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

alias(
    name = "hlo_dce",
    actual = "//xla/hlo/transforms/simplifiers:hlo_dce",
)

alias(
    name = "hlo_memory_scheduler",
    actual = "//xla/hlo/transforms/simplifiers:hlo_memory_scheduler",
)

alias(
    name = "dynamic_dimension_simplifier",
    actual = "//xla/hlo/transforms/simplifiers:dynamic_dimension_simplifier",
)
