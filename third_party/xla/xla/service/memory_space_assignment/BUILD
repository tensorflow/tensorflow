# Description:
#   Memory Space Assignment service implementation.

load(
    "@local_tsl//tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//xla:xla.bzl",
    "xla_cc_test",
)
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

tf_proto_library(
    name = "memory_space_assignment_proto",
    srcs = ["memory_space_assignment.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

cc_library(
    name = "memory_space_assignment",
    srcs = ["memory_space_assignment.cc"],
    hdrs = ["memory_space_assignment.h"],
    deps = [
        ":allocation",
        ":cost_analysis",
        ":memory_bound_loop_optimizer",
        ":memory_space_assignment_proto_cc",
        ":options",
        ":repacking",
        ":slice",
        ":tuning_utils",
        ":utils",
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/service:buffer_value",
        "//xla/service:call_graph",
        "//xla/service:hlo_alias_analysis",
        "//xla/service:hlo_buffer",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_dataflow_analysis",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_value",
        "//xla/service:time_utils",
        "//xla/service/heap_simulator",
        "//xla/service/heap_simulator:allocation_block",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_googlesource_code_re2//:re2",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "memory_space_assignment_test",
    srcs = ["memory_space_assignment_test.cc"],
    deps = [
        ":allocation",
        ":buffer_interval_comparator",
        ":cost_analysis",
        ":memory_space_assignment",
        ":memory_space_assignment_proto_cc",
        ":options",
        ":prefetch_interval_picker",
        ":repacking",
        ":slice",
        ":testing_utils",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_value",
        "//xla/service:instruction_hoister",
        "//xla/service/heap_simulator",
        "//xla/service/heap_simulator:allocation_block",
        "//xla/tests:hlo_test_base",
        "//xla/tests:verified_hlo_module",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "repacking",
    hdrs = ["repacking.h"],
    deps = [
        "//xla:statusor",
        "//xla/service/heap_simulator:allocation_block",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "best_fit_repacker",
    srcs = ["best_fit_repacker.cc"],
    hdrs = ["best_fit_repacker.h"],
    deps = [
        ":repacking",
        "//xla:comparison_util",
        "//xla:statusor",
        "//xla/service/heap_simulator",
        "//xla/service/heap_simulator:allocation_block",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_value",
        "//xla/service/heap_simulator",
    ],
)

cc_library(
    name = "slice",
    srcs = ["slice.cc"],
    hdrs = [
        "slice.h",
    ],
    deps = [
        ":memory_space_assignment_proto_cc",
        "//xla:shape_util",
        "//xla/service:time_utils",
        "//xla/service/heap_simulator",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "slice_test",
    srcs = ["slice_test.cc"],
    deps = [
        ":slice",
        "//xla/service:time_utils",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "allocation",
    srcs = ["allocation.cc"],
    hdrs = [
        "allocation.h",
    ],
    deps = [
        ":cost_analysis",
        ":memory_space_assignment_proto_cc",
        ":slice",
        "//xla:shape_util",
        "//xla:status",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_value",
        "//xla/service:time_utils",
        "//xla/service:tuple_util",
        "//xla/service/heap_simulator",
        "//xla/service/heap_simulator:allocation_block",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "tuning_utils",
    srcs = ["tuning_utils.cc"],
    hdrs = ["tuning_utils.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service/heap_simulator",
    ],
)

cc_library(
    name = "options",
    srcs = [],
    hdrs = ["options.h"],
    deps = [
        ":buffer_interval_comparator",
        ":cost_analysis",
        ":memory_space_assignment_proto_cc",
        ":prefetch_interval_picker",
        ":repacking",
        ":slice",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_value",
        "//xla/service:hlo_value",
        "//xla/service/heap_simulator",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "cost_analysis",
    srcs = ["cost_analysis.cc"],
    hdrs = ["cost_analysis.h"],
    deps = [
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/service:call_graph",
        "//xla/service:hlo_alias_analysis",
        "//xla/service:hlo_buffer",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_value",
        "//xla/service/heap_simulator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "cost_analysis_test",
    srcs = ["cost_analysis_test.cc"],
    deps = [
        ":cost_analysis",
        "//xla:shape_util",
        "//xla:status",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_cost_analysis",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "prefetch_interval_picker",
    srcs = ["prefetch_interval_picker.cc"],
    hdrs = ["prefetch_interval_picker.h"],
    deps = [
        ":cost_analysis",
        ":memory_space_assignment_proto_cc",
        "//xla:shape_util",
        "//xla:status",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_value",
        "//xla/service/heap_simulator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "testing_utils",
    testonly = True,
    hdrs = ["testing_utils.h"],
    deps = [
        ":cost_analysis",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/service:call_graph",
        "//xla/service:hlo_alias_analysis",
        "//xla/service:hlo_cost_analysis",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "memory_bound_loop_optimizer",
    srcs = ["memory_bound_loop_optimizer.cc"],
    hdrs = ["memory_bound_loop_optimizer.h"],
    deps = [
        ":allocation",
        ":cost_analysis",
        ":memory_space_assignment_proto_cc",
        ":options",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/service:buffer_value",
        "//xla/service:hlo_alias_analysis",
        "//xla/service:hlo_buffer",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_value",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "memory_bound_loop_optimizer_test",
    srcs = ["memory_bound_loop_optimizer_test.cc"],
    deps = [
        ":allocation",
        ":cost_analysis",
        ":memory_bound_loop_optimizer",
        ":memory_space_assignment",
        ":memory_space_assignment_proto_cc",
        ":options",
        ":prefetch_interval_picker",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/service:buffer_value",
        "//xla/service:hlo_alias_analysis",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_value",
        "//xla/tests:hlo_test_base",
        "//xla/tests:verified_hlo_module",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@com_googlesource_code_re2//:re2",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "buffer_interval_comparator",
    srcs = ["buffer_interval_comparator.cc"],
    hdrs = ["buffer_interval_comparator.h"],
    deps = [
        ":cost_analysis",
        ":memory_space_assignment_proto_cc",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/service:buffer_value",
        "//xla/service:hlo_value",
        "//xla/service/heap_simulator",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_googlesource_code_re2//:re2",
    ],
)

xla_cc_test(
    name = "prefetch_interval_picker_test",
    srcs = ["prefetch_interval_picker_test.cc"],
    deps = [
        ":cost_analysis",
        ":prefetch_interval_picker",
        ":testing_utils",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_value",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "best_fit_repacker_test",
    srcs = ["best_fit_repacker_test.cc"],
    deps = [
        ":best_fit_repacker",
        ":repacking",
        "//xla:comparison_util",
        "//xla/service/heap_simulator",
        "//xla/service/heap_simulator:allocation_block",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test",
    ],
)
