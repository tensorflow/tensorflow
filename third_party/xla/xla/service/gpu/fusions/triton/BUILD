load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("@local_tsl//tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")
load("//xla:xla.bzl", "xla_cc_test")
load(
    "//xla/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "emitter_helpers",
    srcs = ["emitter_helpers.cc"],
    hdrs = ["emitter_helpers.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:map_mhlo_to_scalar_op",
        "//xla/mlir_hlo:transformation_helpers",
        "//xla/service/gpu:target_util",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@triton//:TritonDialects",
    ],
)

cc_library(
    name = "triton_fusion_emitter",
    srcs = if_gpu_is_configured(
        ["triton_fusion_emitter.cc"],
        ["triton_fusion_emitter_stub.cc"],
    ) + if_cuda_is_configured([
        "compilation_pipeline_cuda.cc",
    ]) + if_rocm_is_configured([
        "compilation_pipeline_rocm.cc",
    ]),
    hdrs = ["triton_fusion_emitter.h"],
    deps = [
        ":emitter_helpers",
        ":passes",
        ":triton_fusion_emitter_legacy_matmul",
        ":triton_support",
        "//xla:autotuning_proto_cc",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_function_importer",
        "//xla/mlir_hlo",
        "//xla/service:dump",
        "//xla/service:hlo_module_config",
        "//xla/service:instruction_fusion",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:hlo_traversal",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/gpu:triton_fusion_analysis",
        "//xla/service/gpu/fusions/ir:xla_gpu",
        "//xla/service/gpu/fusions/mlir:elemental_hlo_to_mlir",
        "//xla/service/gpu/fusions/transforms:passes",
        "//xla/service/gpu/fusions/triton:xla_triton_ops",
        "//xla/service/gpu/llvm_gpu_backend",
        "//xla/service/gpu/model:indexing_analysis",
        "//xla/service/gpu/model:symbolic_tile_analysis",
        "//xla/service/gpu/model:tiled_hlo_instruction_or_computation",
        "//xla/service/gpu/model:triton_emitter_constraints",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:launch_dim",
        "//xla/tools:hlo_decomposer_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:IndexToLLVM",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLToLLVMIRTranslation",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:statusor",
        "@triton//:TritonDialects",
        "@triton//:TritonTransforms",
    ] + if_gpu_is_configured([
        "@triton//:TritonNvidiaGPUTransforms",
        "@triton//:TritonGPUToLLVM",
        "@triton//:TritonToTritonGPU",
        "@triton//:TritonGPUTransforms",
        "@triton//:TritonLLVMIR",
    ]) + if_cuda_is_configured([
        "@triton//third_party/nvidia:NVGPUToLLVM",
        "@triton//third_party/nvidia:TritonNVIDIAGPUToLLVM",
    ]) + if_rocm_is_configured([
        "@local_tsl//tsl/platform:rocm_rocdl_path",
        "@triton//third_party/amd:TritonAMDGPUToLLVM",
        "@triton//third_party/amd:TritonAMDGPUTransforms",
    ]),
)

cc_library(
    name = "triton_fusion_emitter_legacy_matmul",
    srcs = if_gpu_is_configured(
        ["triton_fusion_emitter_legacy_matmul.cc"],
        ["triton_fusion_emitter_legacy_matmul_stub.cc"],
    ),
    hdrs = ["triton_fusion_emitter_legacy_matmul.h"],
    deps = [
        ":emitter_helpers",
        "//xla:comparison_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:map_mhlo_to_scalar_op",
        "//xla/mlir_hlo:transformation_helpers",
        "//xla/service:algorithm_util",
        "//xla/service/gpu:hlo_traversal",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/gpu:target_util",
        "//xla/service/gpu:triton_fusion_analysis",
        "//xla/service/gpu:triton_tiling_propagation",
        "//xla/service/gpu/fusions/triton:xla_triton_ops",
        "//xla/service/gpu/model:tiled_hlo_instruction_or_computation",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:launch_dim",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:tensor_float_32_hdr_lib",
        "@triton//:TritonDialects",
    ],
)

cc_library(
    name = "triton_fusion_emitter_stub_for_testing",
    srcs = [
        "triton_fusion_emitter_legacy_matmul_stub.cc",
        "triton_fusion_emitter_stub.cc",
    ],
    hdrs = [
        "triton_fusion_emitter.h",
        "triton_fusion_emitter_legacy_matmul.h",
    ],
    deps = [
        "//xla:autotuning_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:hlo_traversal",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/gpu:triton_fusion_analysis",
        "//xla/service/gpu/model:tiled_hlo_instruction_or_computation",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:launch_dim",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@triton//:TritonDialects",
    ],
)

xla_cc_test(
    name = "triton_fusion_emitter_stub_test",
    srcs = ["triton_fusion_emitter_stub_test.cc"],
    deps = [
        ":triton_fusion_emitter_stub_for_testing",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:hlo_traversal",
        "//xla/service/gpu/model:tiled_hlo_instruction_or_computation",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_tsl//tsl/platform:test",
    ],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TritonFusionTransforms",
            ],
            "passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    visibility = ["//visibility:private"],
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    srcs = [
        "generalize_kernel_signature.cc",
        "passes.cc",
        "prevent_mmav3_loop_unrolling.cc",
        "sparse_extensions.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":passes_inc_gen",
        ":xla_triton_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:GPUCommonTransforms",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@triton//:TritonAnalysis",
        "@triton//:TritonDialects",
        "@triton//:TritonGPUToLLVM",
        "@triton//:TritonGPUTransforms",
        "@triton//:TritonToTritonGPU",
        "@triton//third_party/nvidia:NVGPUDialect",
        "@triton//third_party/nvidia:NVGPUToLLVM",
        "@triton//third_party/nvidia:TritonNVIDIAGPUToLLVM",
    ],
)

td_library(
    name = "xla_triton_ops_td_files",
    srcs = ["xla_triton_ops.td"],
    includes = ["."],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "xla_triton_ops_inc_gen",
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-op-decls",
                "-dialect=triton_xla",
            ],
            "xla_triton_ops.h.inc",
        ),
        (
            [
                "-gen-op-defs",
                "-dialect=triton_xla",
            ],
            "xla_triton_ops.cc.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=triton_xla",
            ],
            "xla_triton_dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=triton_xla",
            ],
            "xla_triton_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_triton_ops.td",
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@triton//:td_files",
    ],
)

cc_library(
    name = "xla_triton_ops",
    srcs = ["xla_triton_ops.cc"],
    hdrs = ["xla_triton_ops.h"],
    deps = [
        ":xla_triton_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@triton//:TritonDialects",
        "@triton//:triton_gpu_attr_inc_gen",
        "@triton//:triton_gpu_types_inc_gen",
        "@triton//:triton_ops_inc_gen",
    ],
)

xla_test(
    name = "triton_fusion_emitter_device_legacy_test",
    srcs = if_gpu_is_configured(["triton_fusion_emitter_device_legacy_test.cc"]),
    backends = [
        "gpu_a100",
        "gpu_h100",
        "gpu_amd_any",
    ],
    shard_count = 20,
    tags = [
        "no_mac",
    ],
    deps = [
        ":kernel_name_tracer",
        ":triton_fusion_emitter",
        ":triton_test_utils",
        "//xla:autotuning_proto_cc",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu/model:tiled_hlo_instruction_or_computation",
        "//xla/service/gpu/tests:gpu_codegen_test",
        "//xla/stream_executor:device_description",
        "//xla/tests:filecheck",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "triton_fusion_emitter_device_test",
    srcs = if_gpu_is_configured(["triton_fusion_emitter_device_test.cc"]),
    backends = [
        "gpu_a100",
        "gpu_h100",
        "gpu_amd_any",
    ],
    tags = [
        "no_mac",
    ],
    deps = [
        ":triton_fusion_emitter",
        ":triton_test_utils",
        "//xla:autotuning_proto_cc",
        "//xla:error_spec",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu/model:tiled_hlo_instruction_or_computation",
        "//xla/service/gpu/tests:gpu_codegen_test",
        "//xla/stream_executor:device_description",
        "//xla/tests:verified_hlo_module",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "kernel_name_tracer_cuda",
    testonly = True,
    srcs = if_cuda(["kernel_name_tracer_cuda.cc"]),
    hdrs = ["kernel_name_tracer.h"],
    tags = ["manual"],  # Need to exclude this from wildcard builds
    deps = [
        "//xla/backends/profiler/gpu:cupti_collector",
        "//xla/backends/profiler/gpu:cupti_tracer",
        "//xla/tsl/profiler/utils:time_utils",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "kernel_name_tracer_noop",
    testonly = True,
    srcs = ["kernel_name_tracer_noop.cc"],
    hdrs = ["kernel_name_tracer.h"],
    tags = ["manual"],  # Need to exclude this from wildcard builds
)

cc_library(
    name = "kernel_name_tracer",
    testonly = True,
    hdrs = ["kernel_name_tracer.h"],
    deps = if_cuda(
        [":kernel_name_tracer_cuda"],
        [":kernel_name_tracer_noop"],
    ),
)

cc_library(
    name = "triton_test_utils",
    testonly = True,
    srcs = ["triton_test_utils.cc"],
    hdrs = ["triton_test_utils.h"],
    deps = [
        ":triton_fusion_emitter",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:float_normalization",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:gpu_float_support",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/gpu/model:tiled_hlo_instruction_or_computation",
        "//xla/stream_executor:device_description",
        "//xla/tests:filecheck",
        "//xla/tests:hlo_test_base",
        "//xla/tests:verified_hlo_module",
        "//xla/tsl/profiler/utils:time_utils",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "triton_fusion_emitter_mem_utils_test",
    srcs = if_cuda_is_configured(["triton_fusion_emitter_mem_utils_test.cc"]),
    deps = [
        ":triton_fusion_emitter",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:hlo_traversal",
        "//xla/service/gpu/model:symbolic_tile_analysis",
        "//xla/service/gpu/model:tiled_hlo_instruction_or_computation",
        "//xla/service/gpu/model:triton_emitter_constraints",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tests:hlo_test_base",
        "//xla/tests:verified_hlo_module",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:logging",
        "@triton//:TritonDialects",
    ],
)

xla_test(
    name = "triton_fusion_emitter_large_test",
    srcs = if_gpu_is_configured(["triton_fusion_emitter_large_test.cc"]),
    backends = [
        "gpu_a100",
        "gpu_h100",
        "gpu_amd_any",
    ],
    tags = [
        "large",
        "no_mac",
        "no_oss",  # requires-mem:16g tag doesn't work in open source
        "nozapfhahn",  # Times out under coverage
    ] + if_google([
        "requires-mem:16g",
    ]),
    deps = [
        "//xla:error_spec",
        "//xla:xla_proto_cc",
        "//xla/service/gpu/tests:gpu_codegen_test",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "triton_fusion_emitter_parametrized_test",
    srcs = if_gpu_is_configured(["triton_fusion_emitter_parametrized_test.cc"]),
    backends = [
        "gpu_a100",
        "gpu_h100",
        "gpu_amd_any",
    ],
    shard_count = 10,
    tags = ["no_mac"],
    deps = [
        ":triton_support",
        ":triton_test_utils",
        "//xla:comparison_util",
        "//xla:error_spec",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu/tests:gpu_codegen_test",
        "//xla/stream_executor:device_description",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "triton_support",
    srcs = [
        "triton_support.cc",
        "triton_support_legacy.cc",
    ],
    hdrs = [
        "triton_support.h",
        "triton_support_legacy.h",
    ],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:instruction_fusion",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:variant_visitor",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:tensor_float_32_utils",
    ],
)

xla_cc_test(
    name = "triton_support_test",
    srcs = ["triton_support_test.cc"],
    shard_count = 20,
    # TODO(b/353912594): this test does not need to run on GPU, but it is broken on CPU in OSS.
    # Force it to run on GPU temporarily in order to get important OSS coverage.
    tags = ["gpu"],
    deps = [
        ":triton_fusion_emitter",
        ":triton_support",
        ":triton_test_utils",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu/model:tiled_hlo_instruction_or_computation",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "triton_support_legacy_test",
    srcs = if_gpu_is_configured(["triton_support_legacy_test.cc"]),
    backends = [
        "gpu_a100",
        "gpu_h100",
        "gpu_amd_any",
    ],
    tags = ["no_mac"],
    deps = [
        ":kernel_name_tracer",
        ":triton_fusion_emitter",
        ":triton_support",
        ":triton_test_utils",
        "//xla:error_spec",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:triton_fusion_analysis",
        "//xla/service/gpu/model:tiled_hlo_instruction_or_computation",
        "//xla/stream_executor:device_description",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)
