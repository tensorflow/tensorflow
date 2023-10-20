# Automatic sharding annotation

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//xla:xla.bzl", "auto_sharding_deps", "auto_sharding_solver_deps", "xla_cc_binary", "xla_cc_test")

package(
    default_visibility = ["//visibility:public"],
)

package_group(
    name = "friends",
    packages = [
        "//platforms/...",
        "//xla/hlo/experimental/auto_sharding/...",
        "//xla/service/gpu/...",
    ],
)

cc_library(
    name = "auto_sharding",
    srcs = [
        "auto_sharding.cc",
        "auto_sharding_dot_handler.cc",
    ],
    hdrs = [
        "auto_sharding.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":auto_sharding_cost_graph",
        ":auto_sharding_option",
        ":auto_sharding_solver",
        ":auto_sharding_solver_option",
        ":auto_sharding_strategy",
        ":auto_sharding_util",
        ":auto_sharding_wrapper",
        ":cluster_environment",
        ":matrix",
        ":metrics",
        ":profiling_result",
        "//xla:array",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:buffer_value",
        "//xla/service:call_graph",
        "//xla/service:computation_layout",
        "//xla/service:dump",
        "//xla/service:hlo_alias_analysis",
        "//xla/service:hlo_buffer",
        "//xla/service:hlo_memory_scheduler",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_value",
        "//xla/service:sharding_propagation",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ] + auto_sharding_deps(),
)

cc_library(
    name = "auto_sharding_solver_impl",
    srcs = ["auto_sharding_solver_impl.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":auto_sharding_strategy",
        "@com_google_ortools//ortools/linear_solver",
    ],
)

cc_library(
    name = "auto_sharding_solver",
    srcs = ["auto_sharding_solver.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":auto_sharding_strategy",
        "//xla:statusor",
        "//xla:util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_ortools//ortools/linear_solver",
        "@com_google_ortools//ortools/linear_solver:linear_solver_cc_proto",
        "@local_tsl//tsl/platform:hash",
        "@local_tsl//tsl/platform:types",
    ] + auto_sharding_solver_deps(),
)

cc_library(
    name = "auto_sharding_strategy",
    hdrs = [
        "auto_sharding_solver.h",
        "auto_sharding_strategy.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:shape_util",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_value",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_ortools//ortools/linear_solver",
    ],
)

cc_library(
    name = "auto_sharding_cost_graph",
    hdrs = ["auto_sharding_cost_graph.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":auto_sharding_strategy",
        ":matrix",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "auto_sharding_option",
    hdrs = ["auto_sharding_option.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":auto_sharding_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "auto_sharding_wrapper",
    hdrs = ["auto_sharding_wrapper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":auto_sharding_cost_graph",
        ":auto_sharding_strategy",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "auto_sharding_impl",
    srcs = ["auto_sharding_impl.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":auto_sharding_cost_graph",
        ":auto_sharding_option",
        ":auto_sharding_strategy",
        ":auto_sharding_wrapper",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "matrix",
    hdrs = ["matrix.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "cluster_environment",
    srcs = ["cluster_environment.cc"],
    hdrs = ["cluster_environment.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":auto_sharding_solver_option",
        ":auto_sharding_strategy",
        ":auto_sharding_util",
        ":profiling_result",
        "//xla/hlo/ir:hlo",
        "//xla/service/spmd:spmd_partitioner",
    ],
)

cc_library(
    name = "profiling_result",
    hdrs = ["profiling_result.h"],
    visibility = ["//visibility:public"],
    deps = [":auto_sharding_strategy"],
)

cc_library(
    name = "auto_sharding_solver_option",
    hdrs = ["auto_sharding_solver_option.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "auto_sharding_util",
    srcs = ["auto_sharding_util.cc"],
    hdrs = ["auto_sharding_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":auto_sharding_strategy",
        "//xla:array",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:call_graph",
        "//xla/service:sharding_propagation",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    visibility = ["//visibility:public"],
    deps = ["@local_tsl//tsl/lib/monitoring:counter"],
)

xla_cc_binary(
    name = "auto_sharding_runner",
    srcs = ["auto_sharding_runner.cc"],
    deps = [
        ":auto_sharding",
        "//xla:status",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_parser",
        "//xla/stream_executor:dnn",
        "//xla/tools:hlo_module_loader",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

build_test(
    name = "auto_sharding_runner_build_test",
    targets = [
        ":auto_sharding_runner",
    ],
)

xla_cc_test(
    name = "auto_sharding_test",
    srcs = ["auto_sharding_test.cc"],
    deps = [
        ":auto_sharding",
        ":auto_sharding_option",
        ":auto_sharding_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "auto_sharding_solver_test",
    srcs = ["auto_sharding_solver_test.cc"],
    deps = [
        ":auto_sharding_solver",
        ":auto_sharding_strategy",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
    ],
)
