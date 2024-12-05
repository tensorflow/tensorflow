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
    name = "opt_main",
    testonly = True,
    srcs = ["opt_main.cc"],
    deps = [
        ":cpu_opt",
        ":opt_lib",
        "//xla:debug_options_flags",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner",
        "//xla/service:platform_util",
        "//xla/tools:hlo_module_loader",
        "//xla/tools:run_hlo_module_lib",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ] + if_gpu_is_configured([
        ":gpu_opt",
    ]) + if_cuda_is_configured([
        "//xla/stream_executor:cuda_platform",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor:rocm_platform",
    ]),
)

# Includes a macro to register a provider.
cc_library(
    name = "opt_lib",
    srcs = ["opt_lib.cc"],
    hdrs = [
        "opt_lib.h",
        "transforms_example_passes.h",
    ],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms:all_reduce_folder",
        "//xla/hlo/transforms:broadcast_canonicalizer",
        "//xla/hlo/transforms:conditional_canonicalizer",
        "//xla/hlo/transforms:convert_memory_placement_to_internal_annotations",
        "//xla/hlo/transforms:convert_mover",
        "//xla/hlo/transforms:convolution_4d_expander",
        "//xla/hlo/transforms:convolution_pred_expander",
        "//xla/hlo/transforms:dot_decomposer",
        "//xla/hlo/transforms:dynamic_dimension_simplifier",
        "//xla/hlo/transforms:dynamic_index_splitter",
        "//xla/hlo/transforms:eigh_expander",
        "//xla/hlo/transforms:flatten_call_graph",
        "//xla/hlo/transforms:gather_simplifier",
        "//xla/hlo/transforms:hlo_constant_folding",
        "//xla/hlo/transforms:hlo_dce",
        "//xla/hlo/transforms:logistic_expander",
        "//xla/hlo/transforms:optimization_barrier_expander",
        "//xla/hlo/transforms:qr_expander",
        "//xla/hlo/transforms:real_imag_expander",
        "//xla/hlo/transforms:reshape_decomposer",
        "//xla/hlo/transforms:rng_expander",
        "//xla/hlo/transforms:simplify_fp_conversions",
        "//xla/hlo/transforms:slice_sinker",
        "//xla/hlo/transforms:sort_simplifier",
        "//xla/hlo/transforms:stable_sort_expander",
        "//xla/hlo/transforms:stochastic_convert_decomposer",
        "//xla/hlo/transforms:tuple_simplifier",
        "//xla/hlo/transforms:while_loop_trip_count_annotator",
        "//xla/hlo/transforms:zero_sized_hlo_elimination",
        "//xla/hlo/transforms/collectives:all_gather_broadcast_reorder",
        "//xla/hlo/transforms/collectives:all_reduce_contiguous",
        "//xla/hlo/transforms/collectives:collective_quantizer",
        "//xla/service:all_reduce_simplifier",
        "//xla/service:batched_gather_scatter_normalizer",
        "//xla/service:bitcast_dtypes_expander",
        "//xla/service:call_inliner",
        "//xla/service:conditional_simplifier",
        "//xla/service:gather_expander",
        "//xla/service:hlo_graph_dumper",
        "//xla/service:hlo_module_config",
        "//xla/service:platform_util",
        "//xla/service:reduce_scatter_reassociate",
        "//xla/service:scatter_determinism_expander",
        "//xla/service:scatter_simplifier",
        "//xla/service:select_and_scatter_expander",
        "//xla/service:sharding_remover",
        "//xla/service:topk_rewriter",
        "//xla/service:while_loop_constant_sinking",
        "//xla/service:while_loop_simplifier",
        "//xla/service/gpu/transforms:all_gather_dynamic_slice_simplifier",
        "//xla/service/gpu/transforms:all_reduce_splitter",
        "//xla/service/gpu/transforms:collective_permute_valid_iteration_annotator",
        "//xla/service/gpu/transforms:scatter_expander",
        "//xla/service/gpu/transforms:scatter_slice_simplifier",
        "//xla/stream_executor/platform:initialize",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "compiled_opt_lib",
    srcs = ["compiled_opt_lib.cc"],
    hdrs = ["compiled_opt_lib.h"],
    deps = [
        ":opt_lib",
        "//xla:debug_options_flags",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/service:platform_util",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:statusor",
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
        "//xla/hlo/transforms:algebraic_simplifier",
        "//xla/hlo/transforms:reduce_window_rewriter",
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
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:ir_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ] + if_gpu_is_configured([
        "//xla/service:gpu_plugin",
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
        "//xla/hlo/ir:hlo",
        "//xla/service:cpu_plugin",
        "//xla/service:executable",
        "//xla/service:hlo_graph_dumper",
        "//xla/service/cpu:cpu_executable",
        "//xla/stream_executor/host:host_platform",
        "//xla/stream_executor/platform:initialize",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,  # Initializer needs to run.
)

lit_test_suite(
    name = "hlo_opt_tests",
    srcs = enforce_glob(
        [
            "tests/cpu_hlo.hlo",
            "tests/cpu_llvm.hlo",
            "tests/gpu_hlo.hlo",
            "tests/gpu_hlo_backend.hlo",
            "tests/gpu_hlo_buffers.hlo",
            "tests/gpu_hlo_llvm.hlo",
            "tests/gpu_hlo_pass.hlo",
            "tests/gpu_hlo_ptx.hlo",
            "tests/gpu_hlo_unoptimized_llvm.hlo",
            "tests/run_single_pass.hlo",
            "tests/list_passes.hlo",
            "tests/run_multiple_passes.hlo",
            "tests/run_pass_with_input.hlo",
            "tests/gpu_hlo_html.hlo",
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
