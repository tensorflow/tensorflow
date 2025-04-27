load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")
load(
    "//xla/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load("//xla/tsl:tsl.default.bzl", "filegroup")
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

# hlo-opt tool.
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "compiled_opt_lib",
    testonly = True,
    srcs = ["compiled_opt_lib.cc"],
    hdrs = ["compiled_opt_lib.h"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_opt:opt_lib",
        "//xla/hlo/transforms/expanders:bitcast_dtypes_expander",
        "//xla/service:all_reduce_simplifier",
        "//xla/service:all_to_all_decomposer",
        "//xla/service:batched_gather_scatter_normalizer",
        "//xla/service:call_inliner",
        "//xla/service:compiler",
        "//xla/service:conditional_simplifier",
        "//xla/service:conditional_to_select",
        "//xla/service:copy_insertion",
        "//xla/service:executable",
        "//xla/service:gather_expander",
        "//xla/service:map_inliner",
        "//xla/service:platform_util",
        "//xla/service:reduce_scatter_reassociate",
        "//xla/service:scatter_determinism_expander",
        "//xla/service:scatter_simplifier",
        "//xla/service:select_and_scatter_expander",
        "//xla/service:sharding_remover",
        "//xla/service:topk_rewriter",
        "//xla/service:triangular_solve_expander",
        "//xla/service:while_loop_all_reduce_code_motion",
        "//xla/service:while_loop_constant_sinking",
        "//xla/service:while_loop_invariant_code_motion",
        "//xla/service:while_loop_simplifier",
        "//xla/service/gpu/transforms:all_gather_dynamic_slice_simplifier",
        "//xla/service/gpu/transforms:all_reduce_splitter",
        "//xla/service/gpu/transforms:collective_permute_valid_iteration_annotator",
        "//xla/service/gpu/transforms:scatter_expander",
        "//xla/service/gpu/transforms:scatter_slice_simplifier",
        "//xla/service/spmd:sharding_format_picker",
        "//xla/service/spmd/shardy:shardy_xla_pass",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_opt",
    testonly = True,
    srcs = if_gpu_is_configured(["gpu_opt.cc"]),
    tags = ["gpu"],
    deps = [
        ":compiled_opt_lib",
        "//xla:debug_options_flags",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms/simplifiers:algebraic_simplifier",
        "//xla/hlo/transforms/simplifiers:reduce_window_rewriter",
        "//xla/service:buffer_value",
        "//xla/service:compiler",
        "//xla/service:dump",
        "//xla/service:executable",
        "//xla/service:hlo_graph_dumper",
        "//xla/service:hlo_module_config",
        "//xla/service:platform_util",
        "//xla/service/gpu:buffer_sharing",
        "//xla/service/gpu:compile_module_to_llvm_ir",
        "//xla/service/gpu:executable_proto_cc",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/gpu:gpu_hlo_schedule",
        "//xla/service/gpu:gpu_spmd_pipeline",
        "//xla/service/gpu/transforms:all_gather_optimizer",
        "//xla/service/gpu/transforms:cudnn_custom_call_converter",
        "//xla/service/gpu/transforms:dot_algorithm_rewriter",
        "//xla/service/gpu/transforms:dot_dimension_sorter",
        "//xla/service/gpu/transforms:dot_normalizer",
        "//xla/service/gpu/transforms:dot_operand_converter",
        "//xla/service/gpu/transforms:fusion_wrapper",
        "//xla/service/gpu/transforms:gemm_broadcast_folding_rewriter",
        "//xla/service/gpu/transforms:gemm_fusion",
        "//xla/service/gpu/transforms:gemv_rewriter",
        "//xla/service/gpu/transforms:pipelined_p2p_rewriter",
        "//xla/service/gpu/transforms:reduce_scatter_creator",
        "//xla/service/gpu/transforms:reduction_degenerate_dim_remover",
        "//xla/service/gpu/transforms:reduction_dimension_grouper",
        "//xla/service/gpu/transforms:reduction_layout_normalizer",
        "//xla/service/gpu/transforms:rename_fusions",
        "//xla/service/gpu/transforms:sanitize_constant_names",
        "//xla/service/gpu/transforms:stream_attribute_annotator",
        "//xla/service/gpu/transforms:topk_specializer",
        "//xla/service/gpu/transforms:topk_splitter",
        "//xla/service/gpu/transforms:transpose_dimension_grouper",
        "//xla/service/gpu/transforms:windowed_einsum_handler",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/spmd:schedule_aware_collective_ops_cse",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:ir_headers",
    ] + if_gpu_is_configured([
        "//xla/service:gpu_plugin_without_collectives",
        "//xla/service/gpu:gpu_executable",
    ]) + if_cuda_is_configured([
        "//xla/stream_executor:cuda_platform",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor:rocm_platform",
    ]),
    alwayslink = True,  # Initializer needs to run.
)

cc_library(
    name = "cpu_opt",
    testonly = True,
    srcs = ["cpu_opt.cc"],
    deps = [
        ":compiled_opt_lib",
        "//xla:debug_options_flags",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/cpu/codegen:cpu_features",
        "//xla/backends/cpu/codegen:ir_compiler",
        "//xla/backends/cpu/codegen:jit_compiler",
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_opt:opt_lib",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "//xla/service:batchnorm_expander",
        "//xla/service:change_op_data_type",
        "//xla/service:cpu_plugin",
        "//xla/service:dynamic_dimension_inference",
        "//xla/service:dynamic_padder",
        "//xla/service:executable",
        "//xla/service:hlo_execution_profile",
        "//xla/service:hlo_graph_dumper",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_profile_printer_data_cc",
        "//xla/service:hlo_proto_cc",
        "//xla/service:sharding_propagation",
        "//xla/service:transpose_folding",
        "//xla/service/cpu:conv_canonicalization",
        "//xla/service/cpu:cpu_compiler_pure",
        "//xla/service/cpu:cpu_executable",
        "//xla/service/cpu:cpu_instruction_fusion",
        "//xla/service/cpu:cpu_layout_assignment",
        "//xla/service/cpu:dot_op_emitter",
        "//xla/service/cpu:executable_proto_cc",
        "//xla/service/cpu:parallel_task_assignment",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/spmd:stateful_rng_spmd_partitioner",
        "//xla/stream_executor/host:host_platform",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@local_tsl//tsl/platform:platform_port",
    ],
    alwayslink = True,  # Initializer needs to run.
)

lit_test_suite(
    name = "hlo_opt_tests",
    srcs = enforce_glob(
        [
            "tests/cpu_hlo.hlo",
            "tests/cpu_llvm.hlo",
            "tests/cpu_hlo_pass.hlo",
            "tests/gpu_hlo.hlo",
            "tests/gpu_hlo_backend.hlo",
            "tests/gpu_hlo_buffers.hlo",
            "tests/gpu_hlo_collective_cse.hlo",
            "tests/gpu_hlo_llvm.hlo",
            "tests/gpu_hlo_pass.hlo",
            "tests/gpu_hlo_ptx.hlo",
            "tests/gpu_hlo_unoptimized_llvm.hlo",
            "tests/gpu_hlo_html.hlo",
            "tests/list_passes.hlo",
            "tests/run_pass_with_input.hlo",
        ],
        include = [
            "tests/*.hlo",
        ],
    ),
    args = if_cuda_is_configured([
        "--param=PTX=PTX",
        "--param=GPU=a100_pcie_80",
    ]) + if_rocm_is_configured([
        "--param=PTX=GCN",
        "--param=GPU=mi200",
    ]),
    cfg = "//xla:lit.cfg.py",
    data = [":test_utilities"],
    default_tags = tf_cuda_tests_tags(),
    hermetic_cuda_data_dir = "%S/../../../../cuda_nvcc",
    tags_override = {
        "gpu_hlo_ptx.hlo": ["cuda-only"],
    },
    tools = [
        "//xla/tools:hlo-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "gpu_specs/a100_pcie_80.txtpb",
        "gpu_specs/mi200.txtpb",
        "//xla/tools:hlo-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

filegroup(
    name = "all_gpu_specs",
    data = glob(["gpu_specs/*.txtpb"]),
)

exports_files(glob([
    "gpu_specs/*.txtpb",
]))
