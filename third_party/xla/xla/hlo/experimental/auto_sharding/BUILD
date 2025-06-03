# Automatic sharding annotation

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//xla:xla.default.bzl", "xla_cc_binary", "xla_cc_test", "xla_internal")
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_libtpu_portable")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
)

package_group(
    name = "friends",
    packages = xla_internal(["..."]) + [
        "//xla/hlo/experimental/auto_sharding/...",
        "//xla/service/gpu/...",
    ],
)

cc_library(
    name = "auto_sharding",
    srcs = [
        "auto_sharding.cc",
        "auto_sharding_dot_handler.cc",
        "auto_sharding_strategy.cc",
    ],
    hdrs = [
        "auto_sharding.h",
    ],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        ":auto_sharding_cost_graph",
        ":auto_sharding_device_mesh",
        ":auto_sharding_iopddl",
        ":auto_sharding_option",
        ":auto_sharding_solver",
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
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_constant_splitter",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:hlo_memory_scheduler",
        "//xla/hlo/transforms/simplifiers:optimize_input_output_buffer_alias",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:buffer_value",
        "//xla/service:call_graph",
        "//xla/service:computation_layout",
        "//xla/service:dot_as_convolution_util",
        "//xla/service:dump",
        "//xla/service:hlo_buffer",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_value",
        "//xla/service:sharding_propagation",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ] + xla_internal(
        ["experimental/auto_sharding:auto_sharding_impl_internal"],
        otherwise = ["//xla/hlo/experimental/auto_sharding:auto_sharding_impl"],
    ),
)

cc_library(
    name = "auto_sharding_solver_impl",
    srcs = ["auto_sharding_solver_impl.cc"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        ":auto_sharding_proto_cc",
        ":auto_sharding_strategy",
        ":iopddl_lib",
        "@com_google_absl//absl/log:check",
    ] + if_google(
        ["@com_google_ortools//ortools/linear_solver:linear_solver_wrapper"],
        ["@com_google_ortools//ortools/linear_solver"],
    ),
)

cc_library(
    name = "auto_sharding_solver",
    srcs = [
        "auto_sharding_cpsat_for_problem.cc",
        "auto_sharding_cpsat_for_request.cc",
        "auto_sharding_evaluator_for_problem.cc",
        "auto_sharding_evaluator_for_request.cc",
        "auto_sharding_memory.cc",
        "auto_sharding_solver.cc",
    ],
    hdrs = ["auto_sharding_memory.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        ":auto_sharding_device_mesh",
        ":auto_sharding_iopddl",
        ":auto_sharding_proto_cc",
        ":auto_sharding_strategy",
        ":iopddl_lib",
        "//xla:status_macros",
        "//xla:util",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_ortools//ortools/linear_solver:linear_solver_cc_proto",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:hash",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:types",
    ] + xla_internal(
        ["experimental/auto_sharding:auto_sharding_solver_impl_internal"],
        otherwise = ["//xla/hlo/experimental/auto_sharding:auto_sharding_solver_impl"],
    ) + if_google(
        [
            "@com_google_ortools//ortools/linear_solver:linear_solver_sat",
            "@com_google_ortools//ortools/linear_solver:linear_solver_wrapper",
        ],
        ["@com_google_ortools//ortools/linear_solver:linear_solver"],
    ),
)

cc_library(
    name = "auto_sharding_strategy",
    hdrs = [
        "auto_sharding_solver.h",
        "auto_sharding_strategy.h",
    ],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        ":auto_sharding_device_mesh",
        ":auto_sharding_proto_cc",
        ":iopddl_lib",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_value",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ] + if_google(
        ["@com_google_ortools//ortools/linear_solver:linear_solver_wrapper"],
        ["@com_google_ortools//ortools/linear_solver"],
    ),
)

cc_library(
    name = "auto_sharding_cost_graph",
    srcs = ["auto_sharding_cost_graph.cc"],
    hdrs = ["auto_sharding_cost_graph.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        ":auto_sharding_strategy",
        ":matrix",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "auto_sharding_option",
    srcs = ["auto_sharding_option.cc"],
    hdrs = ["auto_sharding_option.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        ":auto_sharding_util",
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "auto_sharding_wrapper",
    hdrs = ["auto_sharding_wrapper.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        ":auto_sharding_cost_graph",
        ":auto_sharding_option",
        ":auto_sharding_strategy",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/service:hlo_cost_analysis",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "auto_sharding_impl",
    srcs = ["auto_sharding_impl.cc"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        ":auto_sharding_cost_graph",
        ":auto_sharding_option",
        ":auto_sharding_strategy",
        ":auto_sharding_wrapper",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/service:hlo_cost_analysis",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "matrix",
    hdrs = ["matrix.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "cluster_environment",
    srcs = ["cluster_environment.cc"],
    hdrs = ["cluster_environment.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        ":auto_sharding_device_mesh",
        ":auto_sharding_option",
        ":auto_sharding_strategy",
        ":auto_sharding_util",
        ":profiling_result",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service/spmd:spmd_partitioner",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "profiling_result",
    hdrs = ["profiling_result.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [":auto_sharding_strategy"],
)

cc_library(
    name = "auto_sharding_util",
    srcs = ["auto_sharding_util.cc"],
    hdrs = ["auto_sharding_util.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        ":auto_sharding_device_mesh",
        ":auto_sharding_strategy",
        "//xla:array",
        "//xla:shape_layout",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:while_loop_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:ptrvec",
        "//xla/hlo/ir:tile_assignment",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:call_graph",
        "//xla/service:computation_layout",
        "//xla/service:sharding_propagation",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@jsoncpp_git//:jsoncpp",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = ["//xla/tsl/lib/monitoring:counter"],
)

xla_cc_binary(
    name = "auto_sharding_runner",
    srcs = ["auto_sharding_runner.cc"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        ":auto_sharding",
        "//xla/hlo/ir:hlo",
        "//xla/tools:hlo_module_loader",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

tf_proto_library(
    name = "auto_sharding_proto",
    srcs = ["auto_sharding.proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "auto_sharding_device_mesh",
    srcs = ["auto_sharding_device_mesh.cc"],
    hdrs = [
        "auto_sharding_device_mesh.h",
    ],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "//xla:array",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
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
    tags = [
        # Disabled until autosharding is fully supported in OSS,
        # https://github.com/openxla/xla/issues/7248.
        "no_oss",
    ],
    deps = [
        ":auto_sharding",
        ":auto_sharding_cost_graph",
        ":auto_sharding_device_mesh",
        ":auto_sharding_option",
        ":auto_sharding_strategy",
        ":auto_sharding_util",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/transforms/simplifiers:hlo_memory_scheduler",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:buffer_value",
        "//xla/service:hlo_value",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "auto_sharding_solver_test",
    srcs = [
        "auto_sharding_cpsat_for_problem_test.cc",
        "auto_sharding_cpsat_for_request_test.cc",
        "auto_sharding_memory_test.cc",
        "auto_sharding_solver_test.cc",
    ],
    tags = [
        # Disabled until autosharding is fully supported in OSS,
        # https://github.com/openxla/xla/issues/7248.
        "no_oss",
    ],
    deps = [
        ":auto_sharding_iopddl",
        ":auto_sharding_proto_cc",
        ":auto_sharding_solver",  # build_cleaner: keep
        ":auto_sharding_strategy",
        ":iopddl_lib",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:statusor",
    ] + if_google(["@com_google_ortools//ortools/linear_solver:linear_solver_scip"]),
)

cc_library(
    name = "iopddl_lib",
    srcs = [
        "iopddl.cc",
        "solver.cc",
    ],
    hdrs = [
        "iopddl.h",
        "solver.h",
    ],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    # CC_TEST_OK=tests imported from the IOPDDL library.
    name = "iopddl_test",
    srcs = ["iopddl_test.cc"],
    deps = [
        ":iopddl_lib",
        "//xla/tsl/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "auto_sharding_iopddl",
    srcs = ["auto_sharding_iopddl.cc"],
    hdrs = ["auto_sharding_iopddl.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        ":iopddl_lib",
        "//xla/hlo/experimental/auto_sharding:auto_sharding_proto_cc",
        "//xla/hlo/experimental/auto_sharding:auto_sharding_strategy",
        "@com_google_absl//absl/log:check",
    ],
)
