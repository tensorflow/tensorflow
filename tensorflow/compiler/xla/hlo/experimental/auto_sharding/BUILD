# Automatic sharding annotation

load("//tensorflow/compiler/xla:xla.bzl", "auto_sharding_deps", "xla_cc_binary", "xla_cc_test")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
)

package_group(
    name = "friends",
    packages = [
        "//platforms/...",
        "//tensorflow/compiler/xla/hlo/experimental/auto_sharding/...",
        "//tensorflow/compiler/xla/service/gpu/...",
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
        "//tensorflow/compiler/xla:array",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_live_range",
        "//tensorflow/compiler/xla/hlo/utils:hlo_sharding_util",
        "//tensorflow/compiler/xla/service:buffer_value",
        "//tensorflow/compiler/xla/service:call_graph",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/service:hlo_alias_analysis",
        "//tensorflow/compiler/xla/service:hlo_buffer",
        "//tensorflow/compiler/xla/service:hlo_memory_scheduler",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_value",
        "//tensorflow/compiler/xla/service:sharding_propagation",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
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
    ] + auto_sharding_deps(),
)

cc_library(
    name = "auto_sharding_solver",
    srcs = ["auto_sharding_solver.cc"],
    hdrs = ["auto_sharding_solver.h"],
    deps = [
        ":auto_sharding_strategy",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/tsl/platform:hash",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_ortools//ortools/linear_solver",
        "@com_google_ortools//ortools/linear_solver:linear_solver_cc_proto",
    ],
)

cc_library(
    name = "auto_sharding_strategy",
    hdrs = ["auto_sharding_strategy.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_value",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "auto_sharding_cost_graph",
    hdrs = ["auto_sharding_cost_graph.h"],
    deps = [
        ":auto_sharding_strategy",
        ":matrix",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "auto_sharding_option",
    hdrs = ["auto_sharding_option.h"],
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
    deps = [
        ":auto_sharding_cost_graph",
        ":auto_sharding_solver",
        ":auto_sharding_strategy",
        "//tensorflow/compiler/xla/hlo/utils:hlo_live_range",
    ],
)

cc_library(
    name = "auto_sharding_impl",
    srcs = ["auto_sharding_impl.cc"],
    deps = [
        ":auto_sharding_cost_graph",
        ":auto_sharding_option",
        ":auto_sharding_solver",
        ":auto_sharding_strategy",
        ":auto_sharding_wrapper",
        "//tensorflow/compiler/xla/hlo/utils:hlo_live_range",
    ],
)

cc_library(
    name = "matrix",
    hdrs = ["matrix.h"],
    deps = [
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cluster_environment",
    srcs = ["cluster_environment.cc"],
    hdrs = ["cluster_environment.h"],
    deps = [
        ":auto_sharding_solver_option",
        ":auto_sharding_strategy",
        ":auto_sharding_util",
        ":profiling_result",
        "//tensorflow/compiler/xla/service/spmd:spmd_partitioner",
    ],
)

cc_library(
    name = "profiling_result",
    hdrs = ["profiling_result.h"],
    deps = [":auto_sharding_strategy"],
)

cc_library(
    name = "auto_sharding_solver_option",
    hdrs = ["auto_sharding_solver_option.h"],
)

cc_library(
    name = "auto_sharding_util",
    srcs = ["auto_sharding_util.cc"],
    hdrs = ["auto_sharding_util.h"],
    deps = [
        ":auto_sharding_strategy",
        "//tensorflow/compiler/xla:array",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_sharding_util",
        "//tensorflow/tsl/platform:status",
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
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = ["//tensorflow/tsl/lib/monitoring:counter"],
)

xla_cc_binary(
    name = "auto_sharding_runner",
    srcs = ["auto_sharding_runner.cc"],
    deps = [
        ":auto_sharding",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/stream_executor:dnn",
        "//tensorflow/compiler/xla/tools:hlo_module_loader",
        "//tensorflow/tsl/platform:platform_port",
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
        ":auto_sharding_util",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "auto_sharding_solver_test",
    srcs = ["auto_sharding_solver_test.cc"],
    deps = [
        ":auto_sharding_solver",
        ":auto_sharding_strategy",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
    ],
)
