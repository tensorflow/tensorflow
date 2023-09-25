# Description:
#   Memory Space Assignment service implementation.

load(
    "//xla:xla.bzl",
    "xla_cc_test",
)
load(
    "@local_tsl//tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
)

cc_library(
    name = "memory_space_assignment",
    srcs = ["memory_space_assignment.cc"],
    hdrs = ["memory_space_assignment.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":memory_space_assignment_proto_cc",
        ":memory_space_assignment_repacking",
        ":memory_space_assignment_tuning_utils",
        ":memory_space_assignment_utils",
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/service:heap_simulator",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_value",
        "//xla/service:tuple_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "memory_space_assignment_test",
    srcs = ["memory_space_assignment_test.cc"],
    deps = [
        ":memory_space_assignment",
        ":memory_space_assignment_proto_cc",
        ":memory_space_assignment_repacking",
        "//xla:shape_util",
        "//xla:status",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:heap_simulator",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_value",
        "//xla/service:instruction_hoister",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "memory_space_assignment_repacking",
    hdrs = ["memory_space_assignment_repacking.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:statusor",
        "//xla:types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "memory_space_assignment_best_fit_repacker",
    srcs = ["memory_space_assignment_best_fit_repacker.cc"],
    hdrs = ["memory_space_assignment_best_fit_repacker.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":memory_space_assignment_repacking",
        "//xla/service:heap_simulator",
    ],
)

cc_library(
    name = "memory_space_assignment_utils",
    srcs = ["memory_space_assignment_utils.cc"],
    hdrs = ["memory_space_assignment_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:heap_simulator",
    ],
)

cc_library(
    name = "memory_space_assignment_tuning_utils",
    srcs = ["memory_space_assignment_tuning_utils.cc"],
    hdrs = ["memory_space_assignment_tuning_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":memory_space_assignment_utils",
        "//xla/hlo/ir:hlo",
        "//xla/service:heap_simulator",
    ],
)

xla_cc_test(
    name = "memory_space_assignment_best_fit_repacker_test",
    srcs = ["memory_space_assignment_best_fit_repacker_test.cc"],
    deps = [
        ":memory_space_assignment_best_fit_repacker",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/platform:test",
    ],
)
