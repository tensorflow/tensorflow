# Tools and utilities that aid in XLA development and usage.

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm", "if_rocm_is_configured")
load(
    "@local_config_sycl//sycl:build_defs.bzl",
    "if_sycl_is_configured",
)
load("//xla:lit.bzl", "lit_test_suite")
load(
    "//xla:xla.default.bzl",
    "xla_cc_binary",
    "xla_cc_test",
    "xla_internal",
    "xla_internal_plugin_deps",
    "xla_py_proto_library",
)
load("//xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl:tsl.bzl",
    "if_cuda_or_rocm",
    "if_google",
)
load("//xla/tsl:tsl.default.bzl", "filegroup", "tsl_pybind_extension")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//xla/tsl/platform/default:build_config_root.bzl",
    "if_llvm_aarch32_available",
    "if_llvm_aarch64_available",
    "if_llvm_hexagon_available",
    "if_llvm_powerpc_available",
    "if_llvm_system_z_available",
    "if_llvm_x86_available",
)
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
    visibility = ["//xla:internal"],
)

#===------------------------------------------------------------------------===#
# XLA command line tools
#===------------------------------------------------------------------------===#

xla_cc_binary(
    name = "hlo-extractor",
    testonly = True,
    deps = [":hlo_extractor_main"],
)

xla_cc_binary(
    name = "hlo-expand",
    testonly = True,
    deps = [":hlo_expand_main"],
)

xla_cc_binary(
    name = "hlo-opt",
    testonly = True,
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lit_lib"],
    deps = [
        "//xla/hlo/tools/hlo_opt:opt_main",
        "//xla/tools/hlo_opt:cpu_opt",
    ] + if_gpu_is_configured([
        "//xla/tools/hlo_opt:gpu_opt",
    ]) + if_cuda_is_configured([
        "//xla/stream_executor:cuda_platform",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor:rocm_platform",
    ]),
)

#===------------------------------------------------------------------------===#
# XLA tools and libraries
#===------------------------------------------------------------------------===#

build_test(
    name = "show_signature_build_test",
    targets = [
        ":show_signature",
    ],
)

xla_cc_binary(
    name = "show_signature",
    srcs = ["show_signature.cc"],
    deps = [
        "//xla:shape_util",
        "//xla/client",
        "//xla/client:client_library",
        "//xla/service:hlo_proto_cc",
        "//xla/service:interpreter_plugin",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:platform_port",
    ],
)

build_test(
    name = "dumped_computation_to_text_build_test",
    targets = [
        ":dumped_computation_to_text",
    ],
)

xla_cc_binary(
    name = "dumped_computation_to_text",
    srcs = ["dumped_computation_to_text.cc"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_proto_cc",
        "//xla/client:client_library",
        "//xla/client:executable_build_options",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/service",
        "//xla/service:hlo_proto_cc",
        "//xla/service:interpreter_plugin",
        "//xla/service:local_service",
        "//xla/tsl/platform:env",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:platform_port",
    ],
)

build_test(
    name = "dumped_computation_to_operation_list_build_test",
    targets = [
        ":dumped_computation_to_operation_list",
    ],
)

xla_cc_binary(
    name = "dumped_computation_to_operation_list",
    srcs = ["dumped_computation_to_operation_list.cc"],
    deps = [
        "//xla:shape_util",
        "//xla/client:client_library",
        "//xla/client:executable_build_options",
        "//xla/client:local_client",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/service",
        "//xla/service:hlo_proto_cc",
        "//xla/service:interpreter_plugin",
        "//xla/service:local_service",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_cc_test(
    name = "hlo_extractor_test",
    srcs = ["hlo_extractor_test.cc"],
    deps = [
        ":hlo_extractor",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_extractor",
    srcs = ["hlo_extractor.cc"],
    hdrs = ["hlo_extractor.h"],
    deps = [
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/simplifiers:algebraic_simplifier",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:call_inliner",
        "//xla/service:compilation_environments",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_verifier",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "hlo_extractor_main",
    srcs = ["hlo_extractor_main.cc"],
    deps = [
        ":hlo_extractor",
        ":hlo_module_loader",
        "//xla:debug_options_flags",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "hlo_expand_main",
    srcs = ["hlo_expand_main.cc"],
    deps = [
        ":hlo_expand_lib",
        ":hlo_module_loader",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "hlo_expand_lib",
    srcs = ["hlo_expand.cc"],
    hdrs = ["hlo_expand.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms/expanders:cholesky_expander",
        "//xla/hlo/transforms/expanders:rng_bit_generator_expander",
        "//xla/hlo/transforms/expanders:rng_expander",
        "//xla/service:batchnorm_expander",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_verifier",
        "//xla/service:sharding_propagation",
        "//xla/service:triangular_solve_expander",
        "//xla/service/spmd:stateful_rng_spmd_partitioner",
        "//xla/tsl/util:command_line_flags",
    ],
)

xla_cc_test(
    name = "hlo_slicer_test",
    srcs = ["hlo_slicer_test.cc"],
    deps = [
        ":hlo_slicer",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_slicer",
    srcs = ["hlo_slicer.cc"],
    hdrs = ["hlo_slicer.h"],
    deps = [
        ":hlo_extractor",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:call_graph",
        "//xla/service:hlo_verifier",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_binary(
    name = "interactive_graphviz",
    srcs = ["interactive_graphviz.cc"],
    deps = [
        ":hlo_extractor",
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:util",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:compiler",
        "//xla/service:cpu_plugin",
        "//xla/service:hlo_graph_dumper",
        "//xla/service:hlo_module_util",
        "//xla/service:hlo_proto_cc",
        "//xla/service:local_service",
        "//xla/service:platform_util",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tools/platform:xprof_loader",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:subprocess",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
    ] + if_cuda_or_rocm(
        [
            "//xla/service:gpu_plugin",
        ],
        xla_internal_plugin_deps(),
    ) + if_cuda([
        "//xla/stream_executor:cuda_platform",
        "@local_config_cuda//cuda:cuda_runtime",
    ]) + if_rocm([
        "//xla/stream_executor:rocm_platform",
    ]) + xla_internal(
        ["service:hlo_graph_dumper_google"],
    ),
)

xla_cc_test(
    name = "interactive_graphviz_bin_test",
    srcs = ["interactive_graphviz_bin_test.cc"],
    data = [
        "data/add.hlo",
        ":interactive_graphviz",
    ],
    deps = [
        "//xla/tsl/platform:subprocess",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_module_loader",
    srcs = ["hlo_module_loader.cc"],
    hdrs = ["hlo_module_loader.h"],
    visibility = ["//xla:friends"],
    deps = [
        ":run_hlo_module_proto_cc",
        "//xla:debug_options_flags",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/translate/mhlo_to_hlo:translate",
        "//xla/hlo/translate/stablehlo_to_hlo:translate",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
        "@com_googlesource_code_re2//:re2",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "hlo_module_loader_test",
    srcs = ["hlo_module_loader_test.cc"],
    deps = [
        ":hlo_module_loader",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "prepare_reference_module",
    srcs = ["prepare_reference_module.cc"],
    hdrs = ["prepare_reference_module.h"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms:despecializer",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner_interface",
        "//xla/stream_executor:platform",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "prepare_reference_module_test",
    srcs = ["prepare_reference_module_test.cc"],
    deps = [
        ":prepare_reference_module",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
    ],
)

tf_proto_library(
    name = "run_hlo_module_proto",
    srcs = ["run_hlo_module.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:xla_data_proto",
    ],
)

xla_py_proto_library(
    name = "run_hlo_module_pb2",
    visibility = ["//visibility:public"],
    deps = [":run_hlo_module_proto"],
)

cc_library(
    name = "hlo_decomposer_lib",
    srcs = ["hlo_decomposer.cc"],
    hdrs = ["hlo_decomposer.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:call_graph",
        "//xla/service:compilation_environments",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "run_hlo_module_lib",
    srcs = ["run_hlo_module.cc"],
    hdrs = ["run_hlo_module.h"],
    deps = [
        ":hlo_control_flow_flattening",
        ":hlo_decomposer_lib",
        ":hlo_module_loader",
        ":prepare_reference_module",
        ":run_hlo_module_proto_cc",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_comparison",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_runner_interface",
        "//xla/service:hlo_verifier",
        "//xla/tests:test_utils",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:path",
    ],
)

xla_cc_test(
    name = "run_hlo_module_test",
    srcs = ["run_hlo_module_test.cc"],
    deps = [
        ":run_hlo_module_lib",
        ":run_hlo_module_proto_cc",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_binary(
    name = "run_hlo_module",
    testonly = True,
    srcs = ["run_hlo_module_main.cc"],
    tags = [
        "noasan",  # Exceeds linker limit.
    ],
    deps = [
        ":run_hlo_module_lib",
        ":run_hlo_module_proto_cc",
        "//xla:debug_options_flags",
        "//xla/service:cpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner",
        "//xla/service:interpreter_plugin",
        "//xla/service:platform_util",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:test",
    ] + if_gpu_is_configured([
        "//xla/service:gpu_plugin",
    ]) + if_cuda([
        "//xla/stream_executor:cuda_platform",
    ]) + if_rocm([
        "//xla/stream_executor:rocm_platform",
    ]) + if_sycl_is_configured([
        "//xla/stream_executor:sycl_platform",
    ]),
)

xla_cc_test(
    name = "run_hlo_module_bin_test",
    srcs = ["run_hlo_module_bin_test.cc"],
    data = [
        "data/add.hlo",
        "data/add_mhlo.mlir",
        "data/add_stablehlo.mlir",
        "data/input_literal_f32_2_2.pbtxt",
        "data/large_constant.hlo",
        "data/must_alias.hlo",
        "data/must_alias_with_sharding.hlo",
        ":run_hlo_module",
    ],
    tags = ["pjrt_migration_candidate"],
    deps = [
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:subprocess",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "hlo_control_flow_flattening",
    srcs = ["hlo_control_flow_flattening.cc"],
    hdrs = ["hlo_control_flow_flattening.h"],
    deps = [
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:call_graph",
        "//xla/service:collective_ops_utils",
        "//xla/service:hlo_proto_cc",
        "//xla/service:tuple_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "hlo_control_flow_flattening_test",
    srcs = ["hlo_control_flow_flattening_test.cc"],
    deps = [
        ":hlo_control_flow_flattening",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/transforms:despecializer",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:collective_ops_utils",
        "//xla/service:hlo_verifier",
        "//xla/service/spmd:spmd_partitioner",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

# This target is used to reproduce miscompiles in OSS outside of TF, and it can
# not have any dependencies apart from the standard library.
cc_library(
    name = "driver",
    srcs = ["driver.cc"],
    tags = if_google(["nofixdeps"]),
    deps = [],
)

cc_library(
    name = "matmul_perf_table_gen",
    srcs = ["matmul_perf_table_gen.cc"],
    hdrs = ["matmul_perf_table_gen.h"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_runner_interface",
        "//xla/service/gpu/model:hlo_op_profile_proto_cc",
        "//xla/service/gpu/model:hlo_op_profiler_lib",
        "//xla/service/gpu/model:hlo_op_profiles",
        "//xla/service/gpu/model:matmul_interpolator_utils",
        "//xla/stream_executor:device_description",
        "//xla/tests:test_utils",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "matmul_perf_table_gen_test",
    srcs = ["matmul_perf_table_gen_test.cc"],
    backends = ["gpu"],
    local_defines = if_cuda(["GOOGLE_CUDA"]),
    deps = [
        ":matmul_perf_table_gen",
        "//xla:xla_data_proto_cc",
        "//xla/service/gpu/model:hlo_op_profile_proto_cc",
        "//xla/service/gpu/tests:hlo_pjrt_gpu_test_base",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "matmul_perf_table_gen_main_lib",
    srcs = ["matmul_perf_table_gen_main.cc"],
    compatible_with = None,
    tags = [
        "no_mac",
    ],
    deps = [
        ":matmul_perf_table_gen",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/target_config",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_allocator_config",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
        "//xla/service:hlo_runner_interface",
        "//xla/service:hlo_runner_pjrt",
        "//xla/service:pjrt_gpu_utils",
        "//xla/service/gpu/model:hlo_op_profile_proto_cc",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/platform:platform_port",
    ] + if_cuda([
        "//xla/stream_executor:cuda_platform",
    ]),
)

cc_library(
    name = "collective_perf_table_gen",
    srcs = ["collective_perf_table_gen.cc"],
    hdrs = ["collective_perf_table_gen.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/target_config",
        "//xla/backends/gpu/transforms/collectives:collective_ops_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/utils:hlo_query",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/service:backend",
        "//xla/service:hlo_module_config",
        "//xla/service:pjrt_gpu_utils",
        "//xla/service/gpu/model:hlo_op_profile_proto_cc",
        "//xla/service/gpu/model:hlo_op_profiles",
        "//xla/tools/multihost_hlo_runner:create_client",
        "//xla/tools/multihost_hlo_runner:functional_hlo_runner",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "collective_perf_table_gen_main_lib",
    srcs = ["collective_perf_table_gen_main.cc"],
    compatible_with = None,
    tags = [
        "no_mac",
    ],
    deps = [
        ":collective_perf_table_gen",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service/gpu/model:collective_interpolator_data",
        "//xla/service/gpu/model:hlo_op_profile_proto_cc",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
    ] + if_cuda([
        "//xla/service:gpu_plugin",
        "//xla/stream_executor:cuda_platform",
    ]),
    alwayslink = True,
)

cc_library(
    name = "compute_xspace_stats",
    srcs = ["compute_xspace_stats.cc"],
    hdrs = ["compute_xspace_stats.h"],
    deps = [
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/profiler/utils:tf_xplane_visitor",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "//xla/tsl/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

xla_cc_test(
    name = "compute_xspace_stats_test",
    srcs = ["compute_xspace_stats_test.cc"],
    deps = [
        ":compute_xspace_stats",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/profiler/utils:group_events",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_test_utils",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

tsl_pybind_extension(
    name = "collective_perf_table_gen_bindings",
    srcs = ["collective_perf_table_gen_bindings.cc"],
    deps = [
        ":collective_perf_table_gen",
        "@com_google_absl//absl/log:check",
        "@nanobind",
    ],
)

xla_test(
    name = "collective_perf_table_gen_test",
    srcs = ["collective_perf_table_gen_test.cc"],
    backends = ["gpu"],
    local_defines = if_cuda(["GOOGLE_CUDA"]),
    deps = [
        ":collective_perf_table_gen",
        "//xla/backends/gpu/transforms/collectives:collective_ops_utils",
        "//xla/service/gpu/model:hlo_op_profile_proto_cc",
        "//xla/service/gpu/tests:hlo_pjrt_gpu_test_base",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_binary(
    name = "collective_perf_table_gen_main",
    tags = [
        "no_mac",
    ],
    deps = [
        ":collective_perf_table_gen_main_lib",
    ],
)

xla_cc_binary(
    name = "matmul_perf_table_gen_main",
    tags = [
        "no_mac",
    ],
    deps = [
        ":matmul_perf_table_gen_main_lib",
    ],
)

xla_test(
    name = "matmul_perf_table_gen_run_test",
    timeout = "eternal",
    srcs = ["matmul_perf_table_gen_run.cc"],
    # TODO(b/372714955): Fix the memory leak.
    args = if_google(["--heap_check="]),
    backends = [
        "h100",
        "b200",
        "amdgpu_any",
    ],
    tags = [
        "gpu",
        "manual",
        "notap",
    ],
    deps = [
        ":matmul_perf_table_gen",
        ":matmul_perf_table_gen_main_lib",
        "//xla/backends/gpu/target_config",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_allocator_config",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
        "//xla/service:hlo_runner_interface",
        "//xla/service:hlo_runner_pjrt",
        "//xla/service:pjrt_gpu_utils",
        "//xla/service/gpu/model:hlo_op_profile_proto_cc",
        "//xla/service/gpu/model:hlo_op_profiles",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
    ],
)

build_test(
    name = "compute_cost_build_test",
    targets = [
        ":compute_cost",
    ],
)

xla_cc_binary(
    name = "compute_cost",
    srcs = ["compute_cost.cc"],
    deps = [
        ":hlo_module_loader",
        "//xla:debug_options_flags",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_cost_analysis",
        "//xla/service/gpu/model:gpu_hlo_cost_analysis",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:platform_port",
    ],
)

lit_test_suite(
    name = "compute_cost_test",
    srcs = ["compute_cost_test.hlo"],
    cfg = "//xla:lit.cfg.py",
    tools = [
        ":compute_cost",
        "@llvm-project//llvm:FileCheck",
    ],
)

xla_cc_binary(
    name = "extract_collective_operations",
    srcs = ["extract_collective_operations.cc"],
    deps = [
        ":hlo_decomposer_lib",
        ":hlo_module_loader",
        "//xla:debug_options_flags",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "llvm_targets",
    deps = [
    ] + if_llvm_aarch32_available([
        "@llvm-project//llvm:ARMAsmParser",  # fixdeps: keep
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
    ]) + if_llvm_aarch64_available([
        "@llvm-project//llvm:AArch64AsmParser",  # fixdeps: keep
        "@llvm-project//llvm:AArch64CodeGen",  # fixdeps: keep
    ]) + if_llvm_hexagon_available([
        "@llvm-project//llvm:HexagonAsmParser",  # fixdeps: keep
        "@llvm-project//llvm:HexagonCodeGen",  # fixdeps: keep
    ]) + if_llvm_powerpc_available([
        "@llvm-project//llvm:PowerPCAsmParser",  # fixdeps: keep
        "@llvm-project//llvm:PowerPCCodeGen",  # fixdeps: keep
    ]) + if_llvm_system_z_available([
        "@llvm-project//llvm:SystemZAsmParser",  # fixdeps: keep
        "@llvm-project//llvm:SystemZCodeGen",  # fixdeps: keep
    ]) + if_llvm_x86_available([
        "@llvm-project//llvm:X86AsmParser",  # fixdeps: keep
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ]),
)

cc_library(
    name = "xla_compile_lib",
    srcs = ["xla_compile_lib.cc"],
    hdrs = ["xla_compile_lib.h"],
    defines = if_llvm_aarch32_available(["XLA_LLVM_AARCH32_AVAILABLE=1"]) + if_llvm_aarch64_available([
        "XLA_LLVM_AARCH64_AVAILABLE=1",
    ]) + if_llvm_hexagon_available([
        "XLA_LLVM_HEXAGON_AVAILABLE=1",
    ]) + if_llvm_powerpc_available([
        "XLA_LLVM_POWERPC_AVAILABLE=1",
    ]) + if_llvm_system_z_available([
        "XLA_LLVM_S390X_AVAILABLE=1",
    ]) + if_llvm_x86_available([
        "XLA_LLVM_X86_AVAILABLE=1",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":hlo_module_loader",
        ":llvm_targets",  # fixdeps: keep
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/backends/cpu:target_machine_options",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/service:export_hlo",
        "//xla/service:gpu_topology",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/service:platform_util",
        "//xla/service:symbol_repository",
        "//xla/service:xla_compile_result_proto_cc_impl",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:cpu_executable",
        "//xla/service/cpu:executable_proto_cc",
        "//xla/service/gpu:gpu_symbol_repository",
        "//xla/service/gpu/autotuning:autotuner_util",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_address_allocator",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_time",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:status_to_from_proto",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_protobuf//:duration_cc_proto",
        "@com_google_protobuf//:protobuf",
        "@llvm-project//llvm:Target",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:register",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:env_time",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:status_to_from_proto",
        "@tsl//tsl/platform:statusor",
    ] + if_cuda_is_configured([
        "//xla/service/gpu:nvptx_compiler",
    ]) + if_rocm_is_configured([
        "//xla/service/gpu:amdgpu_compiler",
    ]),
)

xla_test(
    name = "xla_cpu_compile_lib_test",
    srcs = ["xla_cpu_compile_lib_test.cc"],
    backends = [
        "cpu",
    ],
    data = [
        ":data/add.hlo",
    ],
    deps = [
        ":xla_compile_lib",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:symbol_repository",
        "//xla/service:xla_compile_result_proto_cc",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_time",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "//xla/tsl/protobuf:status_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:duration_cc_proto",
        "@tsl//tsl/platform:path",
    ] + if_cuda([
        "//xla/tsl/cuda:nvml",
    ]),
)

xla_test(
    name = "xla_gpu_compile_lib_test",
    srcs = ["xla_gpu_compile_lib_test.cc"],
    backends = [
        "gpu",
    ],
    data = [
        ":data/add.hlo",
        "//xla/backends/gpu/target_config:all_gpu_specs",
        "//xla/service/gpu:gpu_compiler_test_autotune_db.textproto",
    ],
    deps = [
        ":xla_compile_lib",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:hlo_runner_interface",
        "//xla/service:platform_util",
        "//xla/service:symbol_repository",
        "//xla/service:xla_compile_result_proto_cc_impl",
        "//xla/service/gpu:gpu_symbol_repository",
        "//xla/service/gpu/autotuning:autotuner_util",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "//xla/tsl/protobuf:status_proto_cc",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:path",
    ],
)

# Unlike the tests above, this one doesn't link in any platforms (e.g. cuda_platform).
xla_cc_test(
    name = "xla_deviceless_compile_lib_test",
    srcs = ["xla_deviceless_compile_lib_test.cc"],
    data = [
        "//xla/backends/gpu/target_config:all_gpu_specs",
    ],
    # Although this doesn't require a GPU, we need the tag to prevent this from being
    # run in the OSS CPU presubmits, which are missing the Cuda compile time dependencies.
    tags = ["gpu"],
    deps = [
        ":xla_compile_lib",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/target_config",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:compiler",
        "//xla/service:gpu_plugin",
        "//xla/service:platform_util",
        "//xla/service:symbol_repository",
        "//xla/service:xla_compile_result_proto_cc_impl",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
    ],
)

xla_test(
    name = "hlo_decomposer_test",
    srcs = ["hlo_decomposer_test.cc"],
    deps = [
        ":hlo_decomposer_lib",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_binary(
    name = "print_indexing",
    srcs = ["print_indexing.cc"],
    visibility = ["//visibility:private"],
    deps = [
        ":hlo_module_loader",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:platform_port",
    ],
)

lit_test_suite(
    name = "print_indexing_test",
    srcs = ["print_indexing_test.hlo"],
    cfg = "//xla:lit.cfg.py",
    tools = [
        ":print_indexing",
        "@llvm-project//llvm:FileCheck",
    ],
)

xla_cc_binary(
    name = "compute_xspace_stats_main",
    srcs = ["compute_xspace_stats_main.cc"],
    deps = [
        ":compute_xspace_stats",
        "//xla:debug_options_flags",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_cc_binary(
    name = "compute_xspace_stats_main_gpu",
    srcs = ["compute_xspace_stats_main.cc"],
    tags = ["gpu"],
    deps = [
        ":compute_xspace_stats",
        "//xla:debug_options_flags",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_cc_binary(
    name = "extract_dots_for_benchmark",
    srcs = ["extract_dots_for_benchmark.cc"],
    deps = [
        ":hlo_module_loader",
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_cost_analysis",
        "//xla/service/gpu/model:gpu_hlo_cost_analysis",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:platform_port",
    ],
)
