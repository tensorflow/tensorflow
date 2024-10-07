# Integrates Shardy into the current XLA pipeline.

load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_binary", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        ":friends",
        "//xla:internal",
    ]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//learning/deepmind/jax/ocean/...",
        "//learning/deepmind/partir/...",
        "//platforms/xla/...",
        "//third_party/openxla/shardy/...",
    ],
)

cc_library(
    name = "shardy_xla_pass",
    srcs = ["shardy_xla_pass.cc"],
    hdrs = ["shardy_xla_pass.h"],
    deps = [
        ":constants",
        ":utils",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "//xla/hlo/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/service:computation_layout",
        "//xla/service:hlo_dce",
        "//xla/service:hlo_proto_cc",
        "//xla/service:tuple_simplifier",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/spmd/shardy/mhlo_round_trip:mhlo_export",
        "//xla/service/spmd/shardy/mhlo_round_trip:mhlo_import",
        "//xla/service/spmd/shardy/sdy_round_trip:pipelines",
        "//xla/tsl/framework/mlir:status_scoped_diagnostic_handler",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:statusor",
        "@shardy//shardy/common:file_utils",
        "@shardy//shardy/dialect/sdy/transforms/propagation:passes",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        ":constants",
        "//xla/mlir_hlo",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/ir:register",
    ],
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
    deps = ["@llvm-project//llvm:Support"],
)

xla_cc_test(
    name = "shardy_xla_pass_test",
    srcs = ["shardy_xla_pass_test.cc"],
    deps = [
        ":shardy_xla_pass",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:hlo_test_base",
        "//xla/tests:verified_hlo_module",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_binary(
    name = "sdy_opt",
    srcs = ["sdy_opt_main.cc"],
    deps = [
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/service/spmd/shardy/mhlo_round_trip:export_ops",
        "//xla/service/spmd/shardy/mhlo_round_trip:export_shardings",
        "//xla/service/spmd/shardy/mhlo_round_trip:mhlo_export",
        "//xla/service/spmd/shardy/mhlo_round_trip:mhlo_import",
        "//xla/service/spmd/shardy/mhlo_round_trip:shard_map_export",
        "//xla/service/spmd/shardy/mhlo_round_trip:shard_map_import",
        "//xla/service/spmd/shardy/round_trip_common:convert_sharding_custom_calls",
        "//xla/service/spmd/shardy/round_trip_common:export_named_computations",
        "//xla/service/spmd/shardy/round_trip_common:import_backend_func_calls",
        "//xla/service/spmd/shardy/round_trip_common:import_constants",
        "//xla/service/spmd/shardy/round_trip_common:open_while_free_vars_sharding",
        "//xla/service/spmd/shardy/sdy_round_trip:export_ops",
        "//xla/service/spmd/shardy/sdy_round_trip:export_shardy_attrs",
        "//xla/service/spmd/shardy/sdy_round_trip:import_shardy_attrs",
        "//xla/service/spmd/shardy/sdy_round_trip:pipelines",
        "//xla/service/spmd/shardy/sdy_round_trip:remove_size_one_axes",
        "//xla/service/spmd/shardy/sdy_round_trip:shard_map_export",
        "//xla/service/spmd/shardy/sdy_round_trip:shard_map_import",
        "//xla/service/spmd/shardy/sdy_round_trip/test_utils:mhlo_to_hlo_to_mhlo",
        "//xla/service/spmd/shardy/sdy_round_trip/test_utils:testing_pipeline",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/transforms:passes",
        "@stablehlo//:stablehlo_ops",
    ],
)
