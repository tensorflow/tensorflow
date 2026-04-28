load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@local_config_sycl//sycl:build_defs.bzl",
    "if_sycl_is_configured",
)
load("//xla:lit.bzl", "lit_device_test", "lit_test_suite")
load("//xla:xla.default.bzl", "xla_cc_binary", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google")
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "triton_test_correctness",
    testonly = 1,
    srcs = ["triton_test_correctness.cc"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lit_lib"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:error_spec",
        "//xla/backends/gpu/codegen/tools:test_lib",
        "//xla/codegen/tools:test_lib",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:pjrt_gpu_client_registry",
        "//xla/tools:hlo_module_loader",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest",
    ] + if_cuda_is_configured([
        "//xla/stream_executor/cuda:all_runtime",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:all_runtime",
    ]) + if_sycl_is_configured([
        "//xla/stream_executor/sycl:all_runtime",
    ]),
)

deviceless_tests = ["collectives/all_reduce.hlo"]

lit_test_suite(
    name = "deviceless_tests",
    srcs = deviceless_tests,
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/backends/gpu/codegen/triton/tools:hlo_to_xtileir",
        "@llvm-project//llvm:FileCheck",
    ],
)

lit_device_test(
    name = "tests",
    srcs = glob(
        include = ["**/*.hlo"],
        exclude = deviceless_tests,
    ),
    backends = [
        "h100",
        "b200",
    ],
    data = [
        "//xla/backends/gpu/target_config:all_gpu_specs",
    ],
    tags_override = {
        "dot/dot_unit_tile_sizes.hlo": ["nozapfhahn"],
        "dot/dot.hlo": ["nozapfhahn"],
        "dot/layout_021_201_210.hlo": ["nozapfhahn"],
        "dot/layout_201_012_201.hlo": ["nozapfhahn"],
        "dot/layout_210_210_102.hlo": ["nozapfhahn"],
        "dot/layout_210_210_210.hlo": ["nozapfhahn"],
        "dot/dot_major_lhs_contract.hlo": ["nozapfhahn"],
        "dot/dot_transpose_tma.hlo": ["nozapfhahn"],
        "dot/reshape_of_pad.hlo": ["nozapfhahn"],
        "dot/scaled_dot.hlo": ["nozapfhahn"],
    },
    tools = [
        ":triton_test_correctness",
        "//xla/backends/gpu/codegen/triton/tools:hlo_to_xtileir",
        "//xla/tools:hlo-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

xla_cc_test(
    name = "fusion_emitter_deviceless_test",
    srcs = ["fusion_emitter_deviceless_test.cc"],
    tags = ["no_oss"],  # Doesn't pass in OSS when building with the `fusion_emitter_stub`.
    deps = [
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/codegen/triton:xtile_compiler",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:target_constants",
        "//xla/service/gpu/model:block_level_parameters",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:IR",
    ],
)

xla_test(
    name = "fusion_emitter_device_test",
    srcs = ["fusion_emitter_device_test.cc"],
    backends = [
        "a100",
        "h100",
        "b200",
        "amdgpu_any",
    ],
    shard_count = 10,
    tags = ["no_mac"],
    deps = [
        "//xla:autotuning_proto_cc",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/codegen/triton:support",
        "//xla/backends/gpu/codegen/triton:test_utils",
        "//xla/backends/gpu/codegen/triton:xtile_compiler",
        "//xla/backends/gpu/codegen/triton:xtile_test_base",
        "//xla/backends/gpu/tests:gpu_pjrt_codegen_test",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:algorithm_util",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:target_constants",
        "//xla/service/gpu/model:block_level_parameters",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/rocm:rocm_compute_capability",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@tsl//tsl/platform:path",
    ],
)

xla_test(
    name = "fusion_emitter_large_test",
    srcs = ["fusion_emitter_large_test.cc"],
    backend_tags = {
        "h100": [
            # Needs full H100, otherwise we run OOM.
            "full",
        ],
    },
    backends = [
        "a100",
        "h100",
        "b200",
        "amdgpu_any",
    ],
    shard_count = 3,
    tags = [
        "no_mac",
        "no_oss",  # requires-mem:16g tag doesn't work in open source
        "nozapfhahn",  # Times out under coverage
    ] + if_google([
        "requires-mem:16g",
    ]),
    deps = [
        "//xla:error_spec",
        "//xla:xla_proto_cc",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "fusion_emitter_parametrized_test",
    srcs = ["fusion_emitter_parametrized_test.cc"],
    backends = [
        "a100",
        "h100",
        "b200",
        "amdgpu_any",
    ],
    tags = ["no_mac"],
    use_legacy_runtime = True,
    deps = [
        "//xla:comparison_util",
        "//xla:error_spec",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/codegen/triton:support",
        "//xla/backends/gpu/codegen/triton:test_utils",
        "//xla/backends/gpu/tests:gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "fusion_emitter_shared_dialect_test",
    srcs = ["fusion_emitter_shared_dialect_test.cc"],
    tags = ["no_mac"],
    deps = [
        "//xla/backends/gpu/codegen/triton:xtile_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service/gpu/model:block_level_parameters",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "fusion_emitter_int4_device_test",
    srcs = ["fusion_emitter_int4_device_test.cc"],
    backends = [
        "a100",
        "h100",
        "b200",
        "amdgpu_any",
    ],
    shard_count = 10,
    tags = ["no_mac"],
    deps = [
        "//xla:autotuning_proto_cc",
        "//xla:error_spec",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/tests:hlo_pjrt_gpu_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:device_description",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:path",
    ],
)
