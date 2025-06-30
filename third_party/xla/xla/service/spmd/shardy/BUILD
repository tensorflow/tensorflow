# Integrates Shardy into the current XLA pipeline.

load("//xla:xla.default.bzl", "xla_cc_binary", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "//xla/hlo/translate:stablehlo",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:computation_layout",
        "//xla/service:hlo_proto_cc",
        "//xla/service/spmd/shardy/sdy_round_trip:pipelines",
        "//xla/service/spmd/shardy/stablehlo_round_trip:stablehlo_export",
        "//xla/service/spmd/shardy/stablehlo_round_trip:stablehlo_import",
        "//xla/tsl/framework/mlir:status_scoped_diagnostic_handler",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_googlesource_code_re2//:re2",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:path",
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
        "//xla/service/spmd/shardy/extensions:mhlo_extensions",
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
        "@stablehlo//:stablehlo_ops",
    ],
)

xla_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
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
        ":constants",
        ":shardy_xla_pass",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_binary(
    name = "sdy_opt",
    srcs = ["sdy_opt_main.cc"],
    deps = [
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:stablehlo_extension_passes",
        "//xla/service/spmd/shardy/extensions:mhlo_extensions",
        "//xla/service/spmd/shardy/round_trip_common:export_named_computations",
        "//xla/service/spmd/shardy/round_trip_common:import_constants",
        "//xla/service/spmd/shardy/round_trip_common:import_sdy_custom_calls",
        "//xla/service/spmd/shardy/round_trip_common:import_uninlineable_func_calls",
        "//xla/service/spmd/shardy/round_trip_common:open_while_free_vars_sharding",
        "//xla/service/spmd/shardy/sdy_round_trip:dedup_meshes",
        "//xla/service/spmd/shardy/sdy_round_trip:export_ops",
        "//xla/service/spmd/shardy/sdy_round_trip:export_shardy_attrs",
        "//xla/service/spmd/shardy/sdy_round_trip:import_shardy_attrs",
        "//xla/service/spmd/shardy/sdy_round_trip:pipelines",
        "//xla/service/spmd/shardy/sdy_round_trip:remove_size_one_axes",
        "//xla/service/spmd/shardy/sdy_round_trip:shard_map_export",
        "//xla/service/spmd/shardy/sdy_round_trip:shard_map_import",
        "//xla/service/spmd/shardy/sdy_round_trip/test_utils:stablehlo_to_hlo_to_stablehlo",
        "//xla/service/spmd/shardy/sdy_round_trip/test_utils:testing_pipeline",
        "//xla/service/spmd/shardy/stablehlo_round_trip:export_callback_custom_calls",
        "//xla/service/spmd/shardy/stablehlo_round_trip:export_manual_reduction_collectives_cc",
        "//xla/service/spmd/shardy/stablehlo_round_trip:export_ops",
        "//xla/service/spmd/shardy/stablehlo_round_trip:export_shardings",
        "//xla/service/spmd/shardy/stablehlo_round_trip:shard_map_export",
        "//xla/service/spmd/shardy/stablehlo_round_trip:shard_map_import",
        "//xla/service/spmd/shardy/stablehlo_round_trip:stablehlo_export",
        "//xla/service/spmd/shardy/stablehlo_round_trip:stablehlo_import",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@shardy//shardy/dialect/sdy/ir:register",
        "@shardy//shardy/dialect/sdy/transforms:passes",
        "@shardy//shardy/round_trip_import:pipelines",
    ],
)
