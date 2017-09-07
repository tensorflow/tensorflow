# Description:
#    LLVM-based CPU backend for XLA.

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [":friends"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

load(":build_defs.bzl", "runtime_copts")

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "cpu_compiler",
    srcs = ["cpu_compiler.cc"],
    hdrs = ["cpu_compiler.h"],
    deps = [
        ":compiler_functor",
        ":conv_canonicalization",
        ":cpu_executable",
        ":cpu_instruction_fusion",
        ":cpu_options",
        ":cpu_parallelization_preparation",
        ":disassembler",
        ":ir_emission_utils",
        ":ir_emitter",
        ":layout_assignment",
        ":parallel_cpu_executable",
        ":simple_orc_jit",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:batchnorm_rewriter",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:buffer_liveness",
        "//tensorflow/compiler/xla/service:copy_insertion",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:flatten_call_graph",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_ordering",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_proto",
        "//tensorflow/compiler/xla/service:hlo_proto_util",
        "//tensorflow/compiler/xla/service:hlo_scheduling",
        "//tensorflow/compiler/xla/service:hlo_subcomputation_unification",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service:inliner",
        "//tensorflow/compiler/xla/service:llvm_compiler",
        "//tensorflow/compiler/xla/service:reduce_precision_insertion",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service:transpose_folding",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",  # fixdeps: keep
        "//tensorflow/core:lib",  # fixdeps: keep
        "//tensorflow/core:stream_executor_no_cuda",
        "@llvm//:aarch64_code_gen",  # fixdeps: keep
        "@llvm//:aarch64_disassembler",  # fixdeps: keep
        "@llvm//:arm_code_gen",  # fixdeps: keep
        "@llvm//:arm_disassembler",  # fixdeps: keep
        "@llvm//:core",
        "@llvm//:mc",  # fixdeps: keep
        "@llvm//:object",
        "@llvm//:powerpc_code_gen",  # fixdeps: keep
        "@llvm//:powerpc_disassembler",  # fixdeps: keep
        "@llvm//:support",
        "@llvm//:target",  # fixdeps: keep
        "@llvm//:x86_code_gen",  # fixdeps: keep
        "@llvm//:x86_disassembler",  # fixdeps: keep
    ],
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "simple_orc_jit",
    srcs = ["simple_orc_jit.cc"],
    hdrs = ["simple_orc_jit.h"],
    linkopts = ["-ldl"],
    deps = [
        ":compiler_functor",
        ":cpu_runtime",
        ":cpu_runtime_avx",
        ":cpu_runtime_neon",
        ":cpu_runtime_sse4_1",
        ":disassembler",
        ":runtime_conv2d",
        ":runtime_matmul",
        ":runtime_single_threaded_conv2d",
        ":runtime_single_threaded_matmul",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@llvm//:core",
        "@llvm//:execution_engine",
        "@llvm//:mc",  # fixdeps: keep
        "@llvm//:orc_jit",
        "@llvm//:support",
        "@llvm//:target",  # fixdeps: keep
    ],
)

cc_library(
    name = "cpu_executable",
    srcs = ["cpu_executable.cc"],
    hdrs = ["cpu_executable.h"],
    deps = [
        ":simple_orc_jit",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:device_memory_allocator",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/compiler/xla/service:logical_buffer",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:tuple_points_to_analysis",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "@llvm//:orc_jit",
    ],
)

cc_library(
    name = "parallel_cpu_executable",
    srcs = ["parallel_cpu_executable.cc"],
    hdrs = [
        "parallel_cpu_executable.h",
    ],
    deps = [
        ":cpu_runtime",
        ":shape_partition",
        ":simple_orc_jit",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:device_memory_allocator",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/compiler/xla/service:logical_buffer",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "@llvm//:orc_jit",
    ],
)

cc_library(
    name = "ir_emitter",
    srcs = ["ir_emitter.cc"],
    hdrs = [
        "ir_emitter.h",
    ],
    deps = [
        ":cpu_options",
        ":cpu_runtime",
        ":dot_op_emitter",
        ":elemental_ir_emitter",
        ":ir_emission_utils",
        ":simple_orc_jit",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:elemental_ir_emitter",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:name_uniquer",
        "//tensorflow/compiler/xla/service/llvm_ir:alias_analysis",
        "//tensorflow/compiler/xla/service/llvm_ir:fused_ir_emitter",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_loop",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:loop_emitter",
        "//tensorflow/compiler/xla/service/llvm_ir:ops",
        "//tensorflow/core:lib",
        "@llvm//:core",
        "@llvm//:support",
        "@llvm//:target",
    ],
)

cc_library(
    name = "dot_op_emitter",
    srcs = ["dot_op_emitter.cc"],
    hdrs = ["dot_op_emitter.h"],
    deps = [
        ":cpu_runtime",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_loop",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "@llvm//:core",
    ],
)

cc_binary(
    name = "sample_harness",
    srcs = ["sample_harness.cc"],
    deps = [
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:computation",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "disassembler",
    srcs = ["disassembler.cc"],
    hdrs = ["disassembler.h"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "@llvm//:mc",
        "@llvm//:mc_disassembler",
        "@llvm//:object",
        "@llvm//:powerpc_disassembler",  # fixdeps: keep
        "@llvm//:support",
        "@llvm//:target",
        "@llvm//:x86_disassembler",  # fixdeps: keep
    ],
)

cc_library(
    name = "compiler_functor",
    srcs = ["compiler_functor.cc"],
    hdrs = ["compiler_functor.h"],
    deps = [
        ":cpu_runtime",
        ":cpu_runtime_avx",
        ":cpu_runtime_neon",
        ":cpu_runtime_sse4_1",
        ":disassembler",
        ":llvm_ir_runtime",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:llvm_compiler",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "@llvm//:analysis",
        "@llvm//:core",
        "@llvm//:execution_engine",
        "@llvm//:ipo",
        "@llvm//:mc",
        "@llvm//:object",
        "@llvm//:support",
        "@llvm//:target",
    ],
)

cc_library(
    name = "cpu_runtime_sse4_1",
    srcs = ["cpu_runtime_sse4_1.cc"],
    hdrs = ["cpu_runtime_sse4_1.h"],
    copts = ["-DEIGEN_AVOID_STL_ARRAY"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "cpu_runtime_avx",
    srcs = ["cpu_runtime_avx.cc"],
    hdrs = ["cpu_runtime_avx.h"],
    copts = ["-DEIGEN_AVOID_STL_ARRAY"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "cpu_runtime_neon",
    srcs = ["cpu_runtime_neon.cc"],
    hdrs = ["cpu_runtime_neon.h"],
    # runtime_copts() enables -mfpu=neon
    copts = ["-DEIGEN_AVOID_STL_ARRAY"] + runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "cpu_runtime",
    srcs = [
        "cpu_runtime.cc",
        "xfeed_manager.cc",
    ],
    hdrs = [
        "cpu_runtime.h",
        "xfeed_manager.h",
    ],
    copts = runtime_copts(),
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "llvm_ir_runtime",
    srcs = [
        "llvm_ir_runtime.cc",
    ],
    hdrs = [
        "llvm_ir_runtime.h",
    ],
    deps = [
        "//tensorflow/core:lib",
        "@llvm//:core",
        "@llvm//:transform_utils",
    ],
)

cc_library(
    name = "runtime_conv2d",
    srcs = [
        "runtime_conv2d.cc",
        "runtime_conv2d_impl.h",
    ],
    hdrs = ["runtime_conv2d.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core/kernels:eigen_helpers",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "runtime_matvec",
    srcs = ["runtime_matvec.cc"],
    hdrs = ["runtime_matvec.h"],
    copts = runtime_copts(),
    deps = [
        "//tensorflow/core:framework_lite",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "runtime_matmul",
    srcs = ["runtime_matmul.cc"],
    hdrs = ["runtime_matmul.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_matvec",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/core:framework_lite",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "runtime_single_threaded_conv2d",
    srcs = [
        "runtime_conv2d_impl.h",
        "runtime_single_threaded_conv2d.cc",
    ],
    hdrs = ["runtime_single_threaded_conv2d.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//tensorflow/core/kernels:eigen_helpers",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "runtime_single_threaded_matmul",
    srcs = ["runtime_single_threaded_matmul.cc"],
    hdrs = ["runtime_single_threaded_matmul.h"],
    copts = runtime_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_matvec",
        "//tensorflow/core:framework_lite",
        "//third_party/eigen3",
    ],
)

cc_test(
    name = "cpu_runtime_test",
    srcs = ["cpu_runtime_test.cc"],
    deps = [
        ":cpu_runtime",
        ":runtime_matmul",
        ":runtime_single_threaded_matmul",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//third_party/eigen3",
    ],
)

cc_test(
    name = "cpu_instruction_fusion_test",
    size = "small",
    srcs = ["cpu_instruction_fusion_test.cc"],
    deps = [
        ":cpu_instruction_fusion",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:transpose_folding",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "xfeed_manager_test",
    size = "small",
    srcs = ["xfeed_manager_test.cc"],
    deps = [
        ":cpu_runtime",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "cpu_instruction_fusion",
    srcs = ["cpu_instruction_fusion.cc"],
    hdrs = ["cpu_instruction_fusion.h"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:instruction_fusion",
    ],
)

cc_library(
    name = "cpu_parallelization_preparation",
    srcs = ["cpu_parallelization_preparation.cc"],
    hdrs = [
        "cpu_parallelization_preparation.h",
    ],
    deps = [
        ":ir_emission_utils",
        ":shape_partition",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:logical_buffer",
        "//tensorflow/compiler/xla/service:tuple_points_to_analysis",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "elemental_ir_emitter",
    srcs = ["elemental_ir_emitter.cc"],
    hdrs = ["elemental_ir_emitter.h"],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:elemental_ir_emitter",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "@llvm//:core",
    ],
)

cc_library(
    name = "ir_emission_utils",
    srcs = ["ir_emission_utils.cc"],
    hdrs = ["ir_emission_utils.h"],
    deps = [
        ":cpu_runtime",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla/service:hlo",
    ],
)

cc_library(
    name = "layout_assignment",
    srcs = ["layout_assignment.cc"],
    hdrs = ["layout_assignment.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:layout_assignment",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "conv_canonicalization",
    srcs = ["conv_canonicalization.cc"],
    hdrs = ["conv_canonicalization.h"],
    deps = [
        ":cpu_runtime",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "conv_canonicalization_test",
    size = "small",
    srcs = ["conv_canonicalization_test.cc"],
    deps = [
        ":conv_canonicalization",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
    ],
)

cc_library(
    name = "shape_partition",
    srcs = ["shape_partition.cc"],
    hdrs = ["shape_partition.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
    ],
)

cc_test(
    name = "shape_partition_test",
    srcs = ["shape_partition_test.cc"],
    deps = [
        ":shape_partition",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "cpu_options",
    srcs = ["cpu_options.cc"],
    hdrs = ["cpu_options.h"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_module_config",
    ],
)

# -----------------------------------------------------------------------------

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
