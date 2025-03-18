# Description:
#   Implementation of XLA’s HLO expander transformations.

load("//xla:xla.default.bzl", "xla_cc_test")
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
    name = "op_expander_pass",
    srcs = ["op_expander_pass.cc"],
    hdrs = ["op_expander_pass.h"],
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
    srcs = ["optimization_barrier_expander.cc"],
    hdrs = ["optimization_barrier_expander.h"],
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
    srcs = ["comparison_expander.cc"],
    hdrs = ["comparison_expander.h"],
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
    srcs = ["cholesky_expander.cc"],
    hdrs = ["cholesky_expander.h"],
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
    srcs = ["qr_expander.cc"],
    hdrs = ["qr_expander.h"],
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
    srcs = ["real_imag_expander.cc"],
    hdrs = ["real_imag_expander.h"],
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
    srcs = ["real_imag_expander_test.cc"],
    deps = [
        ":real_imag_expander",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:test",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_creation_utils",
        "//xla/service:pattern_matcher",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "eigh_expander",
    srcs = ["eigh_expander.cc"],
    hdrs = ["eigh_expander.h"],
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
    srcs = ["convolution_4d_expander.cc"],
    hdrs = ["convolution_4d_expander.h"],
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
    srcs = ["convolution_4d_expander_test.cc"],
    deps = [
        "convolution_4d_expander",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "convolution_pred_expander",
    srcs = ["convolution_pred_expander.cc"],
    hdrs = ["convolution_pred_expander.h"],
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
    srcs = ["convolution_pred_expander_test.cc"],
    deps = [
        ":convolution_pred_expander",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/service:pattern_matcher",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "logistic_expander",
    srcs = ["logistic_expander.cc"],
    hdrs = ["logistic_expander.h"],
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
    srcs = ["logistic_expander_test.cc"],
    deps = [
        ":logistic_expander",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:test",
        "//xla/service:dynamic_padder",
        "//xla/service:pattern_matcher",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",  # fixdeps: keep
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "bitcast_dtypes_expander",
    srcs = ["bitcast_dtypes_expander.cc"],
    hdrs = ["bitcast_dtypes_expander.h"],
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
    srcs = ["bitcast_dtypes_expander_test.cc"],
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
    ],
)

cc_library(
    name = "dot_decomposer",
    srcs = ["dot_decomposer.cc"],
    hdrs = ["dot_decomposer.h"],
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
    srcs = ["dot_decomposer_test.cc"],
    deps = [
        ":dot_decomposer",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:pattern_matcher",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",  # fixdeps: keep
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "reshape_decomposer",
    srcs = ["reshape_decomposer.cc"],
    hdrs = ["reshape_decomposer.h"],
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
    srcs = ["reduce_decomposer.cc"],
    hdrs = ["reduce_decomposer.h"],
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
    srcs = ["reduce_decomposer_test.cc"],
    deps = [
        ":reduce_decomposer",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "reshape_decomposer_test",
    srcs = ["reshape_decomposer_test.cc"],
    deps = [
        ":reshape_decomposer",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "stable_sort_expander",
    srcs = ["stable_sort_expander.cc"],
    hdrs = ["stable_sort_expander.h"],
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
    srcs = ["stable_sort_expander_test.cc"],
    deps = [
        ":stable_sort_expander",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:test",
        "//xla/hlo/transforms/simplifiers:algebraic_simplifier",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:pattern_matcher",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "dynamic_index_splitter",
    srcs = ["dynamic_index_splitter.cc"],
    hdrs = ["dynamic_index_splitter.h"],
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
    srcs = ["dynamic_index_splitter_test.cc"],
    deps = [
        ":dynamic_index_splitter",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "rng_expander",
    srcs = ["rng_expander.cc"],
    hdrs = ["rng_expander.h"],
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
    srcs = ["rng_bit_generator_expander.cc"],
    hdrs = ["rng_bit_generator_expander.h"],
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
    name = "stochastic_convert_decomposer",
    srcs = ["stochastic_convert_decomposer.cc"],
    hdrs = ["stochastic_convert_decomposer.h"],
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
    srcs = ["stochastic_convert_decomposer_test.cc"],
    deps = [
        ":stochastic_convert_decomposer",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
