# Description:
#   GPU-specific components in XLA service implementation.

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load(
    "//tensorflow:tensorflow.bzl",
    "check_deps",
    "if_google",
    "tf_cc_test",
    "tf_copts",
    "tf_cuda_library",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "//tensorflow/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load(
    "//tensorflow/core/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "filegroup")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "if_nccl")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

# This target checks that we are not accidentally adding TFRT dependencies.
# It captures the current state of dependencies and might need to get updated
# from time to time.
# Note: private targets may depend on TFRT if they are tagged 'manual'.
check_deps(
    name = "tfrt_deps_check",
    disallowed_deps = select({
        ":is_xlir_enabled": [],
        "//conditions:default": [
            ":xlir_kernels",
            ":xlir_opdefs",
            ":jitrt_custom_calls",
            "@tf_runtime//:basic_kernels_alwayslink",
            "@tf_runtime//:basic_kernels_opdefs",
            "@tf_runtime//:befexecutor",
            "@tf_runtime//:core_runtime",
            "@tf_runtime//:hostcontext",
            "@tf_runtime//:mlirtobef_translate",
            "@tf_runtime//:support",
            "@tf_runtime//:tensor_opdefs",
            "@tf_runtime//:OpBaseTdFiles",
            "@tf_runtime//backends/gpu:gpu_kernels_alwayslink",
            "@tf_runtime//backends/gpu:gpu_opdefs",
            "@tf_runtime//backends/gpu:gpu_types",
            "@tf_runtime//backends/gpu:gpu_wrapper",
            "@tf_runtime//backends/gpu:GpuOpBaseTdFile",
        ],
    }),
    deps = [
        # Targets that are included in CPU builds should not depend on TFRT.
        ":gpu_device_info",
        ":gpu_executable_run_options",
        ":ir_emission_utils",
        ":launch_dimensions",
        ":parallel_loop_emitter",
        ":target_util",
        # XLIR targets should only depend on TFRT if BEF thunk/executable is
        # enabled.
        ":gpu_compiler",
        ":gpu_executable",
        ":nccl_utils",
    ],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

tf_proto_library(
    name = "backend_configs",
    srcs = ["backend_configs.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/stream_executor:dnn_proto",
    ],
)

cc_library(
    name = "gpu_executable_run_options",
    srcs = ["gpu_executable_run_options.cc"],
    hdrs = ["gpu_executable_run_options.h"],
    compatible_with = get_compatible_with_cloud(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:global_device_id",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "gpu_constants",
    srcs = ["gpu_constants.cc"],
    hdrs = ["gpu_constants.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "gpu_types",
    hdrs = ["gpu_types.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/stream_executor:device_description",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "launch_dimensions",
    srcs = [
        "launch_dimensions.cc",
    ],
    hdrs = [
        "launch_dimensions.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":gpu_device_info",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "custom_call_thunk",
    srcs = ["custom_call_thunk.cc"],
    hdrs = ["custom_call_thunk.h"],
    local_defines = if_cuda_is_configured([
        "GOOGLE_CUDA=1",
    ]),
    deps = [
        ":buffer_allocations",
        ":thunk",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:custom_call_status_internal",
        "//tensorflow/core/platform:errors",
        "//tensorflow/stream_executor/gpu:gpu_stream_header",
        "//tensorflow/stream_executor/gpu:gpu_types_header",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "custom_call_test",
    srcs = if_gpu_is_configured(["custom_call_test.cc"]),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/compiler/xla/service:custom_call_status",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/core:test",
        "//tensorflow/stream_executor/gpu:gpu_types_header",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

cc_library(
    name = "stream_assignment",
    srcs = ["stream_assignment.cc"],
    hdrs = ["stream_assignment.h"],
    deps = [
        ":cublas_cudnn",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_reachability",
        "//tensorflow/core/platform:random",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "stream_assignment_test",
    srcs = [
        "stream_assignment_test.cc",
    ],
    tags = ["no_pip"],
    deps = [
        ":stream_assignment",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "hlo_to_ir_bindings",
    srcs = ["hlo_to_ir_bindings.cc"],
    hdrs = ["hlo_to_ir_bindings.h"],
    deps = [
        ":buffer_allocations",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/llvm_ir:buffer_assignment_util",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:tuple_ops",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "target_util",
    srcs = ["target_util.cc"],
    hdrs = ["target_util.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_type_conversion_util",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "gpu_device_info",
    hdrs = ["gpu_device_info.h"],
    compatible_with = get_compatible_with_cloud(),
)

cc_library(
    name = "ir_emitter",
    srcs = [
        "ir_emitter.cc",
        "ir_emitter_nested.cc",
        "ir_emitter_unnested.cc",
    ],
    hdrs = [
        "ir_emitter.h",
        "ir_emitter_context.h",
        "ir_emitter_nested.h",
        "ir_emitter_unnested.h",
        "kernel_mapping_scheme.h",
    ],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]),
    deps = [
        ":backend_configs_cc",
        ":buffer_allocations",
        ":elemental_ir_emitter",
        ":fft_thunk",
        ":gpu_asm_opts_util",
        ":gpu_constants",
        ":gpu_conv_runner",
        ":gpu_executable",
        ":hlo_to_ir_bindings",
        ":ir_emission_utils",
        ":launch_dimensions",
        ":nccl_collective_thunks",
        ":parallel_loop_emitter",
        ":target_util",
        ":thunk",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "//tensorflow/compiler/mlir:name_utils",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/mlir/hlo:lhlo_gpu",
        "//tensorflow/compiler/mlir/xla:attribute_exporter",
        "//tensorflow/compiler/mlir/xla:hlo_module_importer",
        "//tensorflow/compiler/mlir/xla:hlo_utils",
        "//tensorflow/compiler/mlir/xla:mhlo_to_lhlo_with_xla",
        "//tensorflow/compiler/mlir/xla:mlir_hlo_to_hlo",
        "//tensorflow/compiler/mlir/xla:type_to_shape",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:union_find",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:collective_ops_utils",
        "//tensorflow/compiler/xla/service:custom_call_status",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/compiler/xla/service:elemental_ir_emitter",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/compiler/xla/service:name_uniquer",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/service:while_loop_analysis",
        "//tensorflow/compiler/xla/service/llvm_ir:buffer_assignment_util",
        "//tensorflow/compiler/xla/service/llvm_ir:dynamic_update_slice_util",
        "//tensorflow/compiler/xla/service/llvm_ir:fused_ir_emitter",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_builder_mixin",
        "//tensorflow/compiler/xla/service/llvm_ir:kernel_support_library",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_loop",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_type_conversion_util",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:loop_emitter",
        "//tensorflow/compiler/xla/service/llvm_ir:sort_util",
        "//tensorflow/compiler/xla/service/llvm_ir:tuple_ops",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:human_readable_json",
    ] + if_gpu_is_configured([
        ":triangular_solve_thunk",
        ":cholesky_thunk",
    ]),
)

cc_library(
    name = "parallel_loop_emitter",
    srcs = ["parallel_loop_emitter.cc"],
    hdrs = ["parallel_loop_emitter.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":launch_dimensions",
        ":target_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:kernel_support_library",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_loop",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:loop_emitter",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "elemental_ir_emitter",
    srcs = ["elemental_ir_emitter.cc"],
    hdrs = ["elemental_ir_emitter.h"],
    deps = [
        ":backend_configs_cc",
        ":target_util",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:elemental_ir_emitter",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_loop",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service/llvm_ir:loop_emitter",
        "//tensorflow/compiler/xla/service/llvm_ir:math_ops",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "buffer_allocations",
    srcs = ["buffer_allocations.cc"],
    hdrs = ["buffer_allocations.h"],
    deps = [
        ":gpu_constants",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "thunk",
    srcs = ["thunk.cc"],
    hdrs = ["thunk.h"],
    deps = [
        ":buffer_allocations",
        ":gpu_executable_run_options",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
    ],
)

tf_cuda_library(
    name = "nccl_collective_thunks",
    srcs = [
        "nccl_all_gather_thunk.cc",
        "nccl_all_reduce_thunk.cc",
        "nccl_all_to_all_thunk.cc",
        "nccl_collective_permute_thunk.cc",
        "nccl_collective_thunk.cc",
    ],
    hdrs = [
        "nccl_all_gather_thunk.h",
        "nccl_all_reduce_thunk.h",
        "nccl_all_to_all_thunk.h",
        "nccl_collective_permute_thunk.h",
        "nccl_collective_thunk.h",
    ],
    # Override tf_cuda_library()'s internal default value of ["//buildenv/target:gce"].
    compatible_with = [],
    deps = [
        ":buffer_allocations",
        ":ir_emission_utils",
        ":nccl_utils",
        ":thunk",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/mlir/hlo:lhlo_gpu",
        "//tensorflow/compiler/mlir/xla:attribute_exporter",
        "//tensorflow/compiler/mlir/xla:hlo_utils",
        "//tensorflow/compiler/mlir/xla:type_to_shape",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:collective_ops_utils",
        "//tensorflow/compiler/xla/service:global_device_id",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/gpu:gpu_activation_header",
        "//tensorflow/stream_executor/gpu:gpu_stream",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//mlir:IR",
    ],
)

# Empty library to implement nested dependency conditions.
cc_library(name = "empty")

# If NCCL/RCCL is supported, this target '#defines XLA_ENABLE_XCCL' and
# provides a header which #includes NCCL/RCCL.
alias(
    name = "nccl_utils",
    actual = if_nccl(":_nccl_utils", ":empty"),
)

# Do not depend on this target, but rather depend on :nccl_utils.
tf_cuda_library(
    name = "_nccl_utils",
    srcs = if_gpu_is_configured(["nccl_utils.cc"]),
    hdrs = if_gpu_is_configured(["nccl_utils.h"]),
    # Override tf_cuda_library()'s internal default value of ["//buildenv/target:gce"].
    compatible_with = [],
    defines = if_gpu_is_configured(["XLA_ENABLE_XCCL"]),
    tags = ["manual"],  # Only builds with if_nccl().
    deps = if_gpu_is_configured([
        ":gpu_executable_run_options",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:collective_ops_utils",
        "//tensorflow/compiler/xla/service:global_device_id",
        "//tensorflow/compiler/xla/service:rendezvous",
        "//tensorflow/core:lib",
    ]) + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rccl",
    ]),
)

bool_flag(
    name = "enable_xlir",
    build_setting_default = if_google(True, False),
)

config_setting(
    name = "is_xlir_enabled",
    flag_values = {":enable_xlir": "True"},
)

gentbl_cc_library(
    name = "xlir_opdefs_inc_gen",
    tags = ["manual"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "xlir_opdefs.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "xlir_opdefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xlir_ops.td",
    visibility = ["//visibility:private"],
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@tf_runtime//:OpBaseTdFiles",
        "@tf_runtime//backends/gpu:GpuOpBaseTdFile",
    ],
)

cc_library(
    name = "xlir_opdefs",
    srcs = ["xlir_ops.cc"],
    hdrs = ["xlir_ops.h"],
    tags = ["manual"],
    deps = [
        ":gpu_executable_run_options",
        ":io_feed_manager",
        ":xlir_opdefs_inc_gen",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:tensor_opdefs",
        "@tf_runtime//backends/gpu:gpu_opdefs",
        "@tf_runtime//backends/gpu:gpu_wrapper",
    ],
)

cc_library(
    name = "xlir_kernels",
    srcs = ["xlir_kernels.cc"],
    tags = ["manual"],
    visibility = ["//visibility:private"],
    deps = [
        ":nccl_collective_thunks",
        ":nccl_utils",
        ":xlir_opdefs",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:custom_call_status_internal",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//backends/gpu:gpu_kernels_detail",
        "@tf_runtime//backends/gpu:gpu_types",
        "@tf_runtime//backends/gpu:gpu_wrapper",
    ],
    alwayslink = True,  # Contains TFRT kernel registration
)

cc_library(
    name = "jitrt_custom_calls",
    srcs = ["jitrt_custom_calls.cc"],
    hdrs = ["jitrt_custom_calls.h"],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]),
    tags = ["manual"],
    visibility = ["//visibility:private"],
    deps = [
        ":fft_thunk",
        ":gpu_asm_opts_util",
        ":io_feed_manager",
        ":matmul_utils",
        ":nccl_collective_thunks",
        ":stream_executor_util",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//mlir:Support",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:custom_call_status_internal",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:gpu_conv_runner",
        "//tensorflow/core/platform:human_readable_json",
        "//tensorflow/stream_executor/gpu:gpu_stream",
        "//tensorflow/stream_executor/gpu:gpu_types_header",
        "@tf_runtime//:dtype",
        "@tf_runtime//:support",
        "@tf_runtime//backends/jitrt",
        "@tf_runtime//backends/jitrt:custom_call",
    ] + if_gpu_is_configured([
        ":cholesky_thunk",
        ":triangular_solve_thunk",
    ]),
)

cc_library(
    name = "gpu_executable",
    srcs = [
        "bef_thunk.cc",
        "conditional_thunk.cc",
        "convolution_thunk.cc",
        "copy_thunk.cc",
        "for_thunk.cc",
        "gpu_executable.cc",
        "infeed_thunk.cc",
        "kernel_thunk.cc",
        "memset_thunk.cc",
        "outfeed_thunk.cc",
        "replica_id_thunk.cc",
        "sequential_thunk.cc",
        "thunk_schedule.cc",
        "while_thunk.cc",
    ],
    hdrs = [
        "bef_thunk.h",
        "conditional_thunk.h",
        "convolution_thunk.h",
        "copy_thunk.h",
        "custom_call_thunk.h",
        "for_thunk.h",
        "gemm_thunk.h",
        "gpu_executable.h",
        "infeed_thunk.h",
        "kernel_thunk.h",
        "memset_thunk.h",
        "outfeed_thunk.h",
        "replica_id_thunk.h",
        "sequential_thunk.h",
        "thunk_schedule.h",
        "while_thunk.h",
    ],
    local_defines = select({
        ":is_xlir_enabled": ["XLA_ENABLE_XLIR=1"],
        "//conditions:default": [],
    }),
    deps = [
        ":backend_configs_cc",
        ":buffer_allocations",
        ":cusolver_context",
        ":custom_call_thunk",
        ":gemm_thunk",
        ":gpu_asm_opts_util",
        ":gpu_constants",
        ":gpu_conv_runner",
        ":gpu_executable_run_options",
        ":gpu_types",
        ":fft_thunk",
        ":io_feed_manager",
        ":ir_emission_utils",
        ":matmul_utils",
        ":nccl_collective_thunks",
        ":launch_dimensions",
        ":stream_assignment",
        ":stream_executor_util",
        ":thunk",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "//tensorflow/compiler/mlir/hlo:lhlo_gpu",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:refcounting_hash_map",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:custom_call_status_internal",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_dataflow_analysis",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:logical_buffer",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/service:xla_debug_info_manager",
        "//tensorflow/compiler/xla/service/llvm_ir:buffer_assignment_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:scoped_annotation",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor/gpu:asm_compiler",
        "//tensorflow/stream_executor/gpu:gpu_asm_opts",
        "//tensorflow/stream_executor/gpu:gpu_types_header",
        "//tensorflow/stream_executor:blas",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:device_memory_allocator",
        "//tensorflow/stream_executor:kernel",
        "//tensorflow/stream_executor/gpu:gpu_stream",
        "//tensorflow/stream_executor:matmul_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//tensorflow/stream_executor:scratch_allocator",
    ] + if_gpu_is_configured([
        ":cholesky_thunk",
        ":precompiled_kernels",
        ":triangular_solve_thunk",
    ]) + if_cuda_is_configured([
        "//tensorflow/stream_executor/cuda:cuda_stream",
        "//tensorflow/core/platform/default/build_config:cublas_plugin",
        "//tensorflow/core/platform/default/build_config:cudnn_plugin",
        "//tensorflow/core/platform/default/build_config:cufft_plugin",
        "//tensorflow/core/platform/default/build_config:stream_executor_cuda",  # build_cleaner: keep
        "@local_config_cuda//cuda:cuda_headers",
    ]) + if_rocm_is_configured([
        "//tensorflow/core/platform/default/build_config:stream_executor_rocm",
        "@local_config_rocm//rocm:rocm_headers",
    ]) + select({
        ":is_xlir_enabled": [
            ":xlir_kernels",
            ":xlir_opdefs",
            ":jitrt_custom_calls",
            "@llvm-project//llvm:Support",
            "@llvm-project//mlir:GPUTransforms",
            "@llvm-project//mlir:Pass",
            "@llvm-project//mlir:TransformUtils",
            "//tensorflow/compiler/mlir:name_utils",
            "//tensorflow/compiler/mlir/hlo:lhlo",
            "//tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu:lmhlo_to_tfrt_gpu",
            "//tensorflow/compiler/mlir/xla:attribute_exporter",
            "//tensorflow/stream_executor/gpu:gpu_executor_header",
            "@tf_runtime//:basic_kernels_alwayslink",
            "@tf_runtime//:basic_kernels_opdefs",
            "@tf_runtime//:bef",
            "@tf_runtime//backends/jitrt",
            "@tf_runtime//backends/jitrt:jitrt_compiler",
            "@tf_runtime//:befexecutor",
            "@tf_runtime//:beftomlir",
            "@tf_runtime//:core_runtime",
            "@tf_runtime//:hostcontext",
            "@tf_runtime//:init_tfrt_dialects",
            "@tf_runtime//:mlirtobef_translate",
            "@tf_runtime//:support",
            "@tf_runtime//:tensor_alwayslink",
            "@tf_runtime//backends/gpu:gpu_executor",
            "@tf_runtime//backends/gpu:gpu_kernels_alwayslink",
            "@tf_runtime//backends/gpu:gpu_opdefs",
            "@tf_runtime//backends/gpu:gpu_passes",
            "@tf_runtime//backends/gpu:gpu_types",
        ],
        "//conditions:default": [],
    }),
)

# Target used as --@rules_cuda//cuda:cuda_runtime when BefThunk is enabled.
# TFRT uses this target to link against CUDA libraries. Using TFRT's own
# cuda_stubs would duplicate symbols from stream_executor's stubs.
cc_library(
    name = "cuda_runtime_for_xlir",
    tags = ["manual"],  # Build NCCL only if --config=experimental_tfrt_gpu.
    deps = if_cuda_is_configured([
        "//tensorflow/stream_executor/cuda:cublas_lt_stub",
        "//tensorflow/stream_executor/cuda:cublas_stub",
        "//tensorflow/stream_executor/cuda:cuda_stub",
        "//tensorflow/stream_executor/cuda:cudart_stub",
        "//tensorflow/stream_executor/cuda:cudnn_stub",
        "//tensorflow/stream_executor/cuda:cufft_stub",
        "//tensorflow/stream_executor/cuda:cusolver_stub",
        "@local_config_nccl//:nccl",
    ]) + if_rocm_is_configured([
        "@tf_runtime//backends/gpu:cuda_stubs",
    ]),
)

cc_library(
    name = "ir_emission_utils",
    srcs = ["ir_emission_utils.cc"],
    hdrs = ["ir_emission_utils.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":target_util",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/mlir/xla:hlo_utils",
        "//tensorflow/compiler/mlir/xla:type_to_shape",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_type_conversion_util",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "@llvm-project//llvm:Core",
        "@llvm-project//mlir:ArithmeticDialect",
    ],
)

tf_cc_test(
    name = "ir_emission_utils_test",
    srcs = ["ir_emission_utils_test.cc"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:test",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

cc_library(
    name = "cublas_cudnn",
    srcs = ["cublas_cudnn.cc"],
    hdrs = ["cublas_cudnn.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core/platform:statusor",
    ],
)

# TODO(ezhulenev): Extract `RunCholesky` into a separate library.
cc_library(
    name = "cholesky_thunk",
    srcs = if_gpu_is_configured(["cholesky_thunk.cc"]),
    hdrs = if_gpu_is_configured(["cholesky_thunk.h"]),
    deps = if_gpu_is_configured([
        ":buffer_allocations",
        ":cusolver_context",
        ":precompiled_kernels",
        ":thunk",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:stream_header",
        "//tensorflow/stream_executor/gpu:gpu_asm_opts",
    ]),
)

# TODO(ezhulenev): Extract `RunTriangularSolve` into a separate library.
cc_library(
    name = "triangular_solve_thunk",
    srcs = if_gpu_is_configured(["triangular_solve_thunk.cc"]),
    hdrs = if_gpu_is_configured(["triangular_solve_thunk.h"]),
    deps = if_gpu_is_configured([
        ":buffer_allocations",
        ":cusolver_context",
        ":precompiled_kernels",
        ":thunk",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:stream_header",
        "//tensorflow/stream_executor/gpu:gpu_asm_opts",
    ]),
)

cc_library(
    name = "fft_thunk",
    srcs = ["fft_thunk.cc"],
    hdrs = ["fft_thunk.h"],
    deps = [
        ":buffer_allocations",
        ":cusolver_context",
        ":precompiled_kernels",
        ":thunk",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:scratch_allocator",
        "//tensorflow/stream_executor:stream_header",
        "//tensorflow/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "gemm_rewriter",
    srcs = ["gemm_rewriter.cc"],
    hdrs = ["gemm_rewriter.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_evaluator",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
    ],
)

cc_library(
    name = "gemm_thunk",
    srcs = ["gemm_thunk.cc"],
    hdrs = ["gemm_thunk.h"],
    deps = [
        ":matmul_utils",
        ":thunk",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/core/platform:status",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:stream_header",
    ],
)

cc_library(
    name = "gemm_algorithm_picker",
    srcs = if_cuda_is_configured(["gemm_algorithm_picker.cc"]),
    hdrs = if_cuda_is_configured(["gemm_algorithm_picker.h"]),
    deps = if_cuda_is_configured([
        ":backend_configs_cc",
        ":buffer_comparator",
        ":gemm_thunk",
        ":gpu_asm_opts_util",
        ":gpu_conv_runner",
        ":ir_emission_utils",
        ":stream_executor_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/core/protobuf:autotuning_proto_cc",
        "//tensorflow/core/util/proto:proto_utils",
        "//tensorflow/stream_executor:blas",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:device_memory_allocator",
        "//tensorflow/stream_executor:matmul_util",
        "//tensorflow/stream_executor/gpu:redzone_allocator",
    ]) + [":matmul_utils"],
)

cc_library(
    name = "matmul_utils",
    srcs = ["matmul_utils.cc"],
    hdrs = ["matmul_utils.h"],
    deps = [
        ":backend_configs_cc",
        ":ir_emission_utils",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:lhlo_gpu",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/stream_executor:matmul_util",
        "//tensorflow/stream_executor:scratch_allocator",
        "//tensorflow/stream_executor:stream_header",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "matmul_utils_test",
    srcs = ["matmul_utils_test.cc"],
    deps = [
        ":matmul_utils",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
        "//tensorflow/core/platform:status_matchers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_conv_algorithm_picker",
    srcs = ["gpu_conv_algorithm_picker.cc"],
    hdrs = ["gpu_conv_algorithm_picker.h"],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]),
    deps = [
        ":backend_configs_cc",
        ":gpu_asm_opts_util",
        ":gpu_autotuning_proto_cc",
        ":gpu_conv_runner",
        ":gpu_executable",
        ":hlo_algorithm_denylist",
        ":ir_emission_utils",
        ":stream_executor_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core/protobuf:autotuning_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/core/util/proto:proto_utils",
        "//tensorflow/stream_executor:device_memory_allocator",
        "//tensorflow/stream_executor:dnn_proto_cc",
    ] + if_cuda_is_configured([
        ":buffer_comparator",
        "@local_config_cuda//cuda:cudnn_header",
        "//tensorflow/stream_executor/gpu:redzone_allocator",
    ]),
)

cc_library(
    name = "gpu_conv_runner",
    srcs = ["gpu_conv_runner.cc"],
    hdrs = ["gpu_conv_runner.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":stream_executor_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:dnn",
        "//tensorflow/stream_executor:lazy_op_runner",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_conv_rewriter",
    srcs = ["gpu_conv_rewriter.cc"],
    hdrs = ["gpu_conv_rewriter.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "gpu_conv_rewriter_test",
    srcs = ["gpu_conv_rewriter_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cublas_cudnn",
        ":gpu_conv_rewriter",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "cusolver_context",
    srcs = if_gpu_is_configured(["cusolver_context.cc"]),
    hdrs = if_gpu_is_configured(["cusolver_context.h"]),
    deps = [
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:blas",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/stream_executor/cuda:cusolver_lib",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "//tensorflow/stream_executor/rocm:rocblas_wrapper",
        "//tensorflow/stream_executor/rocm:rocsolver_wrapper",
        "//tensorflow/stream_executor/rocm:hipsolver_wrapper",
    ]),
)

cc_library(
    name = "cusolver_rewriter",
    srcs = if_gpu_is_configured(["cusolver_rewriter.cc"]),
    hdrs = if_gpu_is_configured(["cusolver_rewriter.h"]),
    deps = if_gpu_is_configured([
        ":cusolver_context",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:blas",
        "//tensorflow/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/algorithm:container",
    ]),
)

cc_library(
    name = "instruction_fusion",
    srcs = ["instruction_fusion.cc"],
    hdrs = ["instruction_fusion.h"],
    deps = [
        ":gpu_fusible",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:fusion_node_indexing_evaluation",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:instruction_fusion",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service/llvm_ir:fused_ir_emitter",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "instruction_fusion_test",
    srcs = ["instruction_fusion_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":gpu_fusible",
        ":instruction_fusion",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "multi_output_fusion",
    srcs = ["multi_output_fusion.cc"],
    hdrs = ["multi_output_fusion.h"],
    deps = [
        ":gpu_fusible",
        ":instruction_fusion",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_graph_dumper",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_reachability",
        "//tensorflow/compiler/xla/service/llvm_ir:fused_ir_emitter",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "multi_output_fusion_test",
    srcs = ["multi_output_fusion_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":gpu_fusible",
        ":instruction_fusion",
        ":multi_output_fusion",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_sanitize_constant_names",
    srcs = ["gpu_sanitize_constant_names.cc"],
    hdrs = ["gpu_sanitize_constant_names.h"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service/llvm_ir:buffer_assignment_util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "gpu_sanitize_constant_names_test",
    srcs = ["gpu_sanitize_constant_names_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":gpu_sanitize_constant_names",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fusion_bitcast_lift",
    srcs = ["fusion_bitcast_lift.cc"],
    hdrs = ["fusion_bitcast_lift.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "fusion_bitcast_lift_test",
    srcs = ["fusion_bitcast_lift_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":fusion_bitcast_lift",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "fusion_merger",
    srcs = ["fusion_merger.cc"],
    hdrs = ["fusion_merger.h"],
    deps = [
        ":gpu_fusible",
        ":instruction_fusion",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_graph_dumper",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service/llvm_ir:fused_ir_emitter",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "fusion_merger_test",
    srcs = ["fusion_merger_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":fusion_merger",
        ":gpu_fusible",
        ":instruction_fusion",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gpu_conv_padding_legalization",
    srcs = ["gpu_conv_padding_legalization.cc"],
    hdrs = ["gpu_conv_padding_legalization.h"],
    deps = [
        ":cublas_cudnn",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:shape_inference",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "gpu_conv_padding_legalization_test",
    srcs = ["gpu_conv_padding_legalization_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cublas_cudnn",
        ":gpu_conv_padding_legalization",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "cudnn_support_utils",
    srcs = ["cudnn_support_utils.cc"],
    hdrs = ["cudnn_support_utils.h"],
    deps = [
        ":cublas_cudnn",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core/platform:status",
        "//tensorflow/stream_executor:stream_header",
    ],
)

tf_cc_test(
    name = "cudnn_support_utils_test",
    srcs = ["cudnn_support_utils_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cudnn_support_utils",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/stream_executor:device_description",
        "//tensorflow/stream_executor:stream_header",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cudnn_pad_for_convolutions",
    srcs = ["cudnn_pad_for_convolutions.cc"],
    hdrs = ["cudnn_pad_for_convolutions.h"],
    deps = [
        ":cudnn_support_utils",
        ":ir_emission_utils",
        ":stream_executor_util",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/core/platform:status",
        "//tensorflow/stream_executor:stream_header",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

tf_cc_test(
    name = "cudnn_pad_for_convolutions_test",
    srcs = ["cudnn_pad_for_convolutions_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cublas_cudnn",
        ":cudnn_pad_for_convolutions",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "cudnn_vectorize_convolutions",
    srcs = ["cudnn_vectorize_convolutions.cc"],
    hdrs = ["cudnn_vectorize_convolutions.h"],
    deps = [
        ":cudnn_support_utils",
        ":ir_emission_utils",
        ":stream_executor_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/stream_executor:device_description",
        "//tensorflow/stream_executor:stream_header",
    ],
)

tf_cc_test(
    name = "cudnn_vectorize_convolutions_test",
    srcs = ["cudnn_vectorize_convolutions_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cublas_cudnn",
        ":cudnn_vectorize_convolutions",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:call_inliner",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
        "//tensorflow/core/platform:statusor",
    ],
)

cc_library(
    name = "cublas_pad_for_gemms",
    srcs = ["cublas_pad_for_gemms.cc"],
    hdrs = ["cublas_pad_for_gemms.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
    ],
)

tf_cc_test(
    name = "cublas_pad_for_gemms_test",
    srcs = ["cublas_pad_for_gemms_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":cublas_pad_for_gemms",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
    ],
)

cc_library(
    name = "target_constants",
    hdrs = ["target_constants.h"],
)

cc_library(
    name = "gpu_transfer_manager",
    srcs = ["gpu_transfer_manager.cc"],
    hdrs = ["gpu_transfer_manager.h"],
    deps = [
        ":io_feed_manager",
        ":target_constants",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:stream_header",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Core",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "gpu_reduce_scatter_creator",
    srcs = ["gpu_reduce_scatter_creator.cc"],
    hdrs = ["gpu_reduce_scatter_creator.h"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:reduce_scatter_utils",
    ],
)

cc_library(
    name = "gpu_compiler",
    srcs = [
        "gpu_compiler.cc",
    ],
    hdrs = [
        "gpu_compiler.h",
    ],
    local_defines = select({
        ":is_xlir_enabled": ["XLA_ENABLE_XLIR=1"],
        "//conditions:default": [],
    }),
    deps = [
        ":alias_passthrough_params",
        ":all_reduce_blueconnect",
        ":fusion_bitcast_lift",
        ":fusion_merger",
        ":gemm_broadcast_folding_rewriter",
        ":gemm_rewriter",
        ":gpu_constants",
        ":gpu_conv_algorithm_picker",
        ":gpu_conv_rewriter",
        ":gpu_device_info",
        ":gpu_executable",
        ":gpu_hlo_schedule",
        ":gpu_layout_assignment",
        ":gpu_reduce_scatter_creator",
        ":gpu_sanitize_constant_names",
        ":gpu_scatter_expander",
        ":matmul_utils",
        "@llvm-project//mlir:FuncDialect",
        "//tensorflow/compiler/xla/service/spmd:stateful_rng_spmd_partitioner",
        ":gpu_hlo_cost_analysis",
        ":horizontal_input_fusion",
        ":horizontal_loop_fusion",
        ":instruction_fusion",
        ":ir_emission_utils",
        ":ir_emitter",
        ":launch_dimensions",
        ":metrics",
        ":multi_output_fusion",
        ":nccl_collective_thunks",
        ":reduction_degenerate_dim_remover",
        ":reduction_dimension_grouper",
        ":reduction_layout_normalizer",
        ":reduction_splitter",
        ":stream_assignment",
        ":hlo_fusion_stats",
        ":stream_executor_util",
        ":target_constants",
        ":tree_reduction_rewriter",
        ":variadic_op_splitter",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "//tensorflow/compiler/mlir:name_utils",
        "//tensorflow/compiler/mlir/xla:hlo_utils",
        "//tensorflow/compiler/mlir/xla:mhlo_to_lhlo_with_xla",
        "//tensorflow/compiler/mlir/xla:type_to_shape",
        "//tensorflow/compiler/xla/service:bitcast_dtypes_expander",
        "//tensorflow/compiler/xla/service:simplify_fp_conversions",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:all_gather_broadcast_reorder",
        "//tensorflow/compiler/xla/service:dynamic_dimension_simplifier",
        "//tensorflow/compiler/xla/service:all_gather_combiner",
        "//tensorflow/compiler/xla/service:all_gather_decomposer",
        "//tensorflow/compiler/xla/service:all_reduce_combiner",
        "//tensorflow/compiler/xla/service:all_reduce_contiguous",
        "//tensorflow/compiler/xla/service:all_reduce_folder",
        "//tensorflow/compiler/xla/service:all_reduce_reassociate",
        "//tensorflow/compiler/xla/service:all_to_all_decomposer",
        "//tensorflow/compiler/xla/service:async_collective_creator",
        "//tensorflow/compiler/xla/service:batchnorm_expander",
        "//tensorflow/compiler/xla/service:bfloat16_normalization",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:call_inliner",
        "//tensorflow/compiler/xla/service:collectives_schedule_linearizer",
        "//tensorflow/compiler/xla/service:comparison_expander",
        "//tensorflow/compiler/xla/service:conditional_canonicalizer",
        "//tensorflow/compiler/xla/service:conditional_simplifier",
        "//tensorflow/compiler/xla/service:convolution_4d_expander",
        "//tensorflow/compiler/xla/service:copy_insertion",
        "//tensorflow/compiler/xla/service:dot_decomposer",
        "//tensorflow/compiler/xla/service:dot_merger",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/service:dynamic_index_splitter",
        "//tensorflow/compiler/xla/service:dynamic_padder",
        "//tensorflow/compiler/xla/service:eigh_expander",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:flatten_call_graph",
        "//tensorflow/compiler/xla/service:gather_expander",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_dataflow_analysis",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_proto_util",
        "//tensorflow/compiler/xla/service:hlo_subcomputation_unification",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service:llvm_compiler",
        "//tensorflow/compiler/xla/service:logistic_expander",
        "//tensorflow/compiler/xla/service:loop_schedule_linearizer",
        "//tensorflow/compiler/xla/service:optimization_barrier_expander",
        "//tensorflow/compiler/xla/service:operand_upcaster",
        "//tensorflow/compiler/xla/service:qr_expander",
        "//tensorflow/compiler/xla/service:real_imag_expander",
        "//tensorflow/compiler/xla/service:reduce_scatter_combiner",
        "//tensorflow/compiler/xla/service:reshape_decomposer",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service:result_caster",
        "//tensorflow/compiler/xla/service:rng_bit_generator_expander",
        "//tensorflow/compiler/xla/service:rng_expander",
        "//tensorflow/compiler/xla/service:scatter_expander",
        "//tensorflow/compiler/xla/service:sharding_propagation",
        "//tensorflow/compiler/xla/service:sharding_remover",
        "//tensorflow/compiler/xla/service:slice_sinker",
        "//tensorflow/compiler/xla/service:slow_operation_alarm",
        "//tensorflow/compiler/xla/service:sort_simplifier",
        "//tensorflow/compiler/xla/service:stable_sort_expander",
        "//tensorflow/compiler/xla/service:transpose_folding",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/service:while_loop_constant_sinking",
        "//tensorflow/compiler/xla/service:while_loop_simplifier",
        "//tensorflow/compiler/xla/service:while_loop_trip_count_annotator",
        "//tensorflow/compiler/xla/service:zero_sized_hlo_elimination",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/stream_executor:stream_executor_headers",
        ":runtime_intrinsics",
    ] + select({
        ":is_xlir_enabled": [
            "//tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu:pass_utils",
            "@tf_runtime//:mlirtobef_translate",
            "@tf_runtime//:support",
            "@tf_runtime//:bef",
            "@tf_runtime//backends/gpu:gpu_executor",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "nvptx_compiler",
    srcs = if_cuda_is_configured([
        "nvptx_compiler_registration.cc",
    ]),
    deps = if_cuda_is_configured([
        ":nvptx_compiler_impl",
    ]),
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "nvptx_helper",
    srcs = ["nvptx_helper.cc"],
    hdrs = ["nvptx_helper.h"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:cuda_libdevice_path",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "nvptx_compiler_impl",
    srcs = if_cuda_is_configured([
        "nvptx_compiler.cc",
    ]),
    hdrs = if_cuda_is_configured([
        "nvptx_compiler.h",
    ]),
    deps = if_cuda_is_configured([
        ":cublas_pad_for_gemms",
        ":cudnn_fused_conv_rewriter",
        ":cudnn_pad_for_convolutions",
        ":cudnn_vectorize_convolutions",
        ":cusolver_rewriter",
        ":gemm_algorithm_picker",
        ":gpu_asm_opts_util",
        ":gpu_executable",
        ":gpu_compiler",
        ":gpu_conv_padding_legalization",
        ":gpu_conv_rewriter",
        ":gpu_layout_assignment",
        ":ir_emission_utils",
        ":nvptx_helper",
        ":target_constants",
        ":triangular_solve_rewriter",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:node_hash_map",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:call_inliner",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service:llvm_compiler",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:cuda_libdevice_path",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/stream_executor/cuda:cuda_diagnostics",
        "//tensorflow/stream_executor/gpu:asm_compiler",
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor/gpu:gpu_driver_header",
    ]) + [
        ":metrics",
    ],
)

tf_cc_test(
    name = "nvptx_compiler_test",
    srcs = if_gpu_is_configured([
        "nvptx_compiler_test.cc",
    ]),
    tags = [
        "gpu",
        "no_rocm",
        "nomsan",  # Pulls in precompiled NVIDIA libraries which cause false
        # positives in msan.
    ],
    deps = [
        ":nvptx_compiler_impl",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
    ],
)

tf_cc_test(
    name = "gpu_aot_compilation_test",
    srcs = [
        "gpu_aot_compilation_test.cc",
    ],
    tags = [
        "gpu",
        "manual",  # Requires --//tensorflow/compiler/xla/service/gpu:enable_bef_executable=true
        "no_oss",
        "no_rocm",
        "nomsan",  # Pulls in precompiled NVIDIA libraries which cause false
        "notap",
        # positives in msan.
        "requires-gpu-nvidia",
    ],
    deps = [
        ":nvptx_compiler_impl",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
    ],
)

cc_library(
    name = "amdgpu_compiler",
    srcs = if_rocm_is_configured([
        "amdgpu_compiler_registration.cc",
    ]),
    deps = if_rocm_is_configured([
        ":amdgpu_compiler_impl",
    ]),
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "amdgpu_compiler_impl",
    srcs = if_rocm_is_configured([
        "amdgpu_compiler.cc",
    ]),
    hdrs = if_rocm_is_configured([
        "amdgpu_compiler.h",
    ]),
    deps = if_rocm_is_configured([
        ":cusolver_rewriter",
        ":gemm_rewriter",
        ":gpu_compiler",
        ":gpu_conv_algorithm_picker",
        ":gpu_conv_padding_legalization",
        ":gpu_conv_rewriter",
        ":gpu_layout_assignment",
        ":reduction_degenerate_dim_remover",
        ":reduction_dimension_grouper",
        ":reduction_layout_normalizer",
        ":target_constants",
        ":tree_reduction_rewriter",
        ":triangular_solve_rewriter",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:call_inliner",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core/platform:rocm_rocdl_path",
    ]),
)

cc_library(
    name = "all_reduce_blueconnect",
    srcs = ["all_reduce_blueconnect.cc"],
    hdrs = ["all_reduce_blueconnect.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_query",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "all_reduce_blueconnect_test",
    srcs = ["all_reduce_blueconnect_test.cc"],
    deps = [
        ":all_reduce_blueconnect",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:status_matchers",
    ],
)

cc_library(
    name = "xfeed_queue",
    hdrs = ["xfeed_queue.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "io_feed_manager",
    srcs = [
        "infeed_manager.cc",
        "outfeed_manager.cc",
        "xla_executor_state.h",
    ],
    hdrs = [
        "infeed_manager.h",
        "outfeed_manager.h",
    ],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]),
    deps = [
        ":xfeed_queue",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor/gpu:gpu_executor_header",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "gpu_layout_assignment",
    srcs = ["gpu_layout_assignment.cc"],
    hdrs = ["gpu_layout_assignment.h"],
    deps = [
        ":backend_configs_cc",
        ":ir_emission_utils",
        ":matmul_utils",
        ":stream_executor_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:layout_assignment",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/core/platform:tensor_float_32_hdr_lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "gpu_layout_assignment_test",
    srcs = ["gpu_layout_assignment_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":cublas_cudnn",
        ":gemm_rewriter",
        ":gpu_layout_assignment",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # build_cleaner: keep
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_hlo_schedule",
    srcs = ["gpu_hlo_schedule.cc"],
    hdrs = ["gpu_hlo_schedule.h"],
    deps = [
        ":stream_assignment",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:buffer_value",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_memory_scheduler",
        "//tensorflow/compiler/xla/service:hlo_ordering",
        "//tensorflow/compiler/xla/service:hlo_reachability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "gpu_hlo_schedule_test",
    srcs = [
        "gpu_hlo_schedule_test.cc",
    ],
    tags = ["no_pip"],
    deps = [
        ":gpu_hlo_schedule",
        ":stream_assignment",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "while_transformer_test",
    srcs = ["while_transformer_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":instruction_fusion",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/service:copy_insertion",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service:while_loop_analysis",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "stream_executor_util",
    srcs = ["stream_executor_util.cc"],
    hdrs = ["stream_executor_util.h"],
    copts = tf_copts(),
    deps = [
        ":cublas_cudnn",
        ":launch_dimensions",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:cuda_libdevice_path",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/protobuf:autotuning_proto_cc",
        "//tensorflow/core/util:determinism_for_kernels",
        "//tensorflow/core/util/proto:proto_utils",
        "//tensorflow/stream_executor:kernel_spec",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gpu_asm_opts_util",
    srcs = ["gpu_asm_opts_util.cc"],
    hdrs = ["gpu_asm_opts_util.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/stream_executor/gpu:gpu_asm_opts",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_hlo_cost_analysis",
    srcs = ["gpu_hlo_cost_analysis.cc"],
    hdrs = ["gpu_hlo_cost_analysis.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
    ],
)

tf_cc_test(
    name = "gpu_hlo_cost_analysis_test",
    srcs = ["gpu_hlo_cost_analysis_test.cc"],
    deps = [
        ":gpu_hlo_cost_analysis",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "buffer_comparator",
    srcs = if_cuda_is_configured(["buffer_comparator.cc"]),
    hdrs = if_cuda_is_configured(["buffer_comparator.h"]),
    deps = if_cuda_is_configured([
        ":launch_dimensions",
        ":gpu_asm_opts_util",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:stream_executor_headers",
        "//tensorflow/stream_executor/gpu:asm_compiler",
    ]),
)

tf_cc_test(
    name = "buffer_comparator_test",
    srcs = if_cuda_is_configured(["buffer_comparator_test.cc"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/core:test_main",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:test",
    ] + if_cuda_is_configured([
        ":buffer_comparator",
        "//tensorflow/core/platform/default/build_config:stream_executor_cuda",  # build_cleaner: keep
        "//tensorflow/stream_executor:device_memory",
    ]),
)

cc_library(
    name = "gpu_fusible",
    srcs = ["gpu_fusible.cc"],
    hdrs = ["gpu_fusible.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:instruction_fusion",
    ],
)

tf_cc_test(
    name = "gpu_fusible_test",
    srcs = ["gpu_fusible_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":gpu_fusible",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cudnn_fused_conv_rewriter",
    srcs = ["cudnn_fused_conv_rewriter.cc"],
    hdrs = ["cudnn_fused_conv_rewriter.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/stream_executor:dnn_proto_cc",
    ],
)

tf_cc_test(
    name = "cudnn_fused_conv_rewriter_test",
    srcs = ["cudnn_fused_conv_rewriter_test.cc"],
    tags = [
        "gpu",
        "no_oss",
        "noasan",
        "nomsan",
        "requires-gpu-sm70",
    ],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        ":cudnn_fused_conv_rewriter",
        ":gpu_conv_rewriter",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/service:pattern_matcher_gmock",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_codegen_test",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "variadic_op_splitter",
    srcs = ["variadic_op_splitter.cc"],
    hdrs = ["variadic_op_splitter.h"],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gpu_scatter_expander",
    srcs = ["gpu_scatter_expander.cc"],
    hdrs = ["gpu_scatter_expander.h"],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:scatter_expander",
        "@com_google_absl//absl/algorithm:container",
    ],
)

tf_cc_test(
    name = "variadic_op_splitter_test",
    srcs = ["variadic_op_splitter_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":ir_emission_utils",
        ":variadic_op_splitter",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_proto_library(
    name = "gpu_autotuning_proto",
    srcs = ["gpu_autotuning.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo_proto",
        "//tensorflow/core/protobuf:autotuning_proto",
    ],
)

cc_library(
    name = "hlo_algorithm_denylist",
    srcs = ["hlo_algorithm_denylist.cc"],
    hdrs = ["hlo_algorithm_denylist.h"],
    deps = [
        ":gpu_autotuning_proto_cc",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/core/protobuf:autotuning_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "hlo_algorithm_denylist_test",
    srcs = ["hlo_algorithm_denylist_test.cc"],
    data = ["data/hlo_algorithm_denylist.pbtxt"],
    tags = ["no_pip"],
    deps = [
        ":hlo_algorithm_denylist",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/stream_executor:dnn",
    ],
)

cc_library(
    name = "alias_passthrough_params",
    srcs = ["alias_passthrough_params.cc"],
    hdrs = ["alias_passthrough_params.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
    ],
)

tf_cc_test(
    name = "alias_passthrough_params_test",
    srcs = ["alias_passthrough_params_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":alias_passthrough_params",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "horizontal_loop_fusion",
    srcs = ["horizontal_loop_fusion.cc"],
    hdrs = ["horizontal_loop_fusion.h"],
    deps = [
        ":gpu_fusible",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "horizontal_loop_fusion_test",
    srcs = ["horizontal_loop_fusion_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":fusion_merger",
        ":horizontal_loop_fusion",
        ":instruction_fusion",
        ":multi_output_fusion",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:tuple_simplifier",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "horizontal_input_fusion",
    srcs = ["horizontal_input_fusion.cc"],
    hdrs = ["horizontal_input_fusion.h"],
    deps = [
        ":gpu_fusible",
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "horizontal_input_fusion_test",
    srcs = ["horizontal_input_fusion_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":horizontal_input_fusion",
        ":multi_output_fusion",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_codegen_test",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "reduction_degenerate_dim_remover",
    srcs = ["reduction_degenerate_dim_remover.cc"],
    hdrs = ["reduction_degenerate_dim_remover.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "reduction_dimension_grouper",
    srcs = ["reduction_dimension_grouper.cc"],
    hdrs = ["reduction_dimension_grouper.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "reduction_splitter",
    srcs = ["reduction_splitter.cc"],
    hdrs = ["reduction_splitter.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
    ],
)

tf_cc_test(
    name = "reduction_splitter_test",
    srcs = ["reduction_splitter_test.cc"],
    deps = [
        ":reduction_splitter",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/service:hlo_matchers",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "reduction_layout_normalizer",
    srcs = ["reduction_layout_normalizer.cc"],
    hdrs = ["reduction_layout_normalizer.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tree_reduction_rewriter",
    srcs = ["tree_reduction_rewriter.cc"],
    hdrs = ["tree_reduction_rewriter.h"],
    deps = [
        ":ir_emission_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/service:collective_ops_utils",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gemm_broadcast_folding_rewriter",
    srcs = ["gemm_broadcast_folding_rewriter.cc"],
    hdrs = ["gemm_broadcast_folding_rewriter.h"],
    deps = [
        ":backend_configs_cc",
        ":cublas_cudnn",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:pattern_matcher",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/stream_executor/lib",
    ],
)

# These tests are intended to be run with --test_env=XLA_FLAGS=--xla_gpu_bef_thunk
# See tap/tensorflow.xla_gpu_tfrt.
# TODO(b/187959015): Expand test patterns; see cl/377552996.
test_suite(
    name = "bef_thunk_tests",
    tests = [
        # Disabled: Repeated Kernel BefThunk invocations prolong test duration.
        # E.g., it takes ~1300s for 256x128 matrix.
        # "//tensorflow/compiler/xla/client/lib:svd_test_gpu",
        "//tensorflow/compiler/xla/service/gpu:cudnn_fused_conv_rewriter_test",
        "//tensorflow/compiler/xla/service/gpu:custom_call_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gemm_broadcast_folding_rewrite_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_alignment_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_copy_test",
        "//tensorflow/compiler/xla/service/gpu/tests:kernel_launch_test",
        "//tensorflow/compiler/xla/service/gpu/tests:mlir_gemm_test",
        "//tensorflow/compiler/xla/tests:cholesky_test_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_cudnn_frontend_disabled_gpu",
        "//tensorflow/compiler/xla/tests:dot_operation_test_gpu",
        "//tensorflow/compiler/xla/tests:local_client_execute_test_gpu",
        "//tensorflow/compiler/xla/tests:multioutput_fusion_test_gpu",
        "//tensorflow/compiler/xla/tests:outfeed_in_nested_computation_test_gpu",
        "//tensorflow/compiler/xla/tests:scatter_test_gpu",
        "//tensorflow/compiler/xla/tests:select_and_scatter_test_gpu",
        "//tensorflow/compiler/xla/tests:tuple_test_gpu",
        "//tensorflow/compiler/xla/tests:while_test_gpu",
    ] + if_google([
        # Currently fails in OSS.
        "//tensorflow/compiler/xla/tests:convolution_test_gpu",
    ]),
)

# These tests are intended to be run with --test_env=XLA_FLAGS=--xla_gpu_bef_executable
# See tap/tensorflow.xla_gpu_tfrt_executable.
test_suite(
    name = "bef_executable_tests",
    tests = [
        "//tensorflow/compiler/tests:fft_test_gpu",
        "//tensorflow/compiler/xla/service/gpu:cudnn_fused_conv_rewriter_test",
        "//tensorflow/compiler/xla/service/gpu:custom_call_test",
        "//tensorflow/compiler/xla/service/gpu/tests:add_preds.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:all_reduce.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:bef_executable_test_gpu",
        "//tensorflow/compiler/xla/service/gpu/tests:concat.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:constant.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:copy.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:copy_nested.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:dynamic_update_slice_inplace.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:element_wise_row_vectorization.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:element_wise_row_vectorization_test",
        "//tensorflow/compiler/xla/service/gpu/tests:fused_scatter.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:fused_slice.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:fused_slice_different_operands.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:fusion.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:gemm_broadcast_folding_rewrite_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_alignment_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_atomic_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_copy_alone_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_copy_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_ftz_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_fusion_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_index_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_infeed_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_input_fusible_slice_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_kernel_tiling_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_ldg_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_noalias_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_reduce_scatter_creator_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_spmd_e2e_compile_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_too_many_blocks_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_unrolling_test",
        "//tensorflow/compiler/xla/service/gpu/tests:kernel_launch_test",
        "//tensorflow/compiler/xla/service/gpu/tests:launch_dimensions.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:mlir_fft_test",
        "//tensorflow/compiler/xla/service/gpu/tests:mlir_gemm_test",
        "//tensorflow/compiler/xla/service/gpu/tests:mlir_gpu_compile_test",
        "//tensorflow/compiler/xla/service/gpu/tests:pad_to_static.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:parallel_reduction_test",
        "//tensorflow/compiler/xla/service/gpu/tests:pred_arithmetic_test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduce_unnested.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduction_degenerate_dim_remover_test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduction_dimension_grouper_test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduction_layout_normalizer_test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduction_vectorization_sm_all.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduction_vectorization_test",
        "//tensorflow/compiler/xla/service/gpu/tests:rng_get_and_update_state.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:scatter.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:select_and_scatter.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:slice_to_dynamic.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:sorting.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:sorting_test",
        "//tensorflow/compiler/xla/service/gpu/tests:tree_reduction_rewriter_test",
        "//tensorflow/compiler/xla/tests:array_elementwise_ops_test_gpu",
        "//tensorflow/compiler/xla/tests:axpy_simple_test_gpu",
        "//tensorflow/compiler/xla/tests:bad_rng_shape_validation_test_gpu",
        "//tensorflow/compiler/xla/tests:batch_normalization_test_gpu",
        "//tensorflow/compiler/xla/tests:bfloat16_test_gpu",
        "//tensorflow/compiler/xla/tests:binop_scaling_test_gpu",
        "//tensorflow/compiler/xla/tests:bitcast_convert_test_gpu",
        "//tensorflow/compiler/xla/tests:broadcast_simple_test_gpu",
        "//tensorflow/compiler/xla/tests:broadcast_test_gpu",
        "//tensorflow/compiler/xla/tests:call_test_gpu",
        "//tensorflow/compiler/xla/tests:check_execution_arity_test_gpu",
        "//tensorflow/compiler/xla/tests:cholesky_test_gpu",
        "//tensorflow/compiler/xla/tests:client_test_gpu",
        "//tensorflow/compiler/xla/tests:compilation_cache_test_gpu",
        "//tensorflow/compiler/xla/tests:compute_constant_test_gpu",
        "//tensorflow/compiler/xla/tests:concat_test_gpu",
        "//tensorflow/compiler/xla/tests:constant_reduction_function_test_gpu",
        "//tensorflow/compiler/xla/tests:constants_test_gpu",
        "//tensorflow/compiler/xla/tests:convert_test_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_1d_autotune_disabled_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_autotune_disabled_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_cudnn_frontend_disabled_gpu",
        "//tensorflow/compiler/xla/tests:copy_test_gpu",
        "//tensorflow/compiler/xla/tests:cpu_gpu_fusion_test_gpu",
        "//tensorflow/compiler/xla/tests:deallocation_test_gpu",
        "//tensorflow/compiler/xla/tests:deconstruct_tuple_test_gpu",
        "//tensorflow/compiler/xla/tests:deep_graph_test_gpu",
        "//tensorflow/compiler/xla/tests:dot_operation_single_threaded_runtime_test_gpu",
        "//tensorflow/compiler/xla/tests:dot_operation_test_autotune_disabled_gpu",
        "//tensorflow/compiler/xla/tests:dot_operation_test_gpu",
        "//tensorflow/compiler/xla/tests:dynamic_ops_test_gpu",
        "//tensorflow/compiler/xla/tests:execution_profile_test_gpu",
        "//tensorflow/compiler/xla/tests:execution_profile_test_with_xla_hlo_profile_gpu",
        "//tensorflow/compiler/xla/tests:exhaustive_binary_16_bit_test_gpu",
        "//tensorflow/compiler/xla/tests:exhaustive_binary_test_f32_f64_gpu",
        "//tensorflow/compiler/xla/tests:exhaustive_unary_test_complex_gpu",
        "//tensorflow/compiler/xla/tests:exhaustive_unary_test_f32_or_smaller_gpu",
        "//tensorflow/compiler/xla/tests:exhaustive_unary_test_f64_gpu",
        "//tensorflow/compiler/xla/tests:floor_ceil_test_gpu",
        "//tensorflow/compiler/xla/tests:fmax_fmin_test_gpu",
        "//tensorflow/compiler/xla/tests:gather_operation_test_gpu",
        "//tensorflow/compiler/xla/tests:get_dimension_size_test_gpu",
        "//tensorflow/compiler/xla/tests:half_test_gpu",
        "//tensorflow/compiler/xla/tests:iota_test_gpu",
        "//tensorflow/compiler/xla/tests:local_client_allocation_test_gpu",
        "//tensorflow/compiler/xla/tests:local_client_execute_test_gpu",
        "//tensorflow/compiler/xla/tests:log_test_gpu",
        "//tensorflow/compiler/xla/tests:map_test_gpu",
        "//tensorflow/compiler/xla/tests:matrix_ops_simple_test_gpu",
        "//tensorflow/compiler/xla/tests:multidimensional_slice_test_gpu",
        "//tensorflow/compiler/xla/tests:multioutput_fusion_test_gpu",
        "//tensorflow/compiler/xla/tests:pad_test_gpu",
        "//tensorflow/compiler/xla/tests:params_test_gpu",
        "//tensorflow/compiler/xla/tests:pred_test_gpu",
        "//tensorflow/compiler/xla/tests:prng_test_gpu",
        "//tensorflow/compiler/xla/tests:ptxas_bug_120501638_gpu",
        "//tensorflow/compiler/xla/tests:query_inferred_shape_test_gpu",
        "//tensorflow/compiler/xla/tests:reduce_hlo_test_gpu",
        "//tensorflow/compiler/xla/tests:reduce_precision_test_gpu",
        "//tensorflow/compiler/xla/tests:replay_test_gpu",
        "//tensorflow/compiler/xla/tests:reshape_motion_test_gpu",
        "//tensorflow/compiler/xla/tests:reshape_test_gpu",
        "//tensorflow/compiler/xla/tests:reverse_test_gpu",
        "//tensorflow/compiler/xla/tests:round_trip_packed_literal_test_gpu",
        "//tensorflow/compiler/xla/tests:round_trip_transfer_test_gpu",
        "//tensorflow/compiler/xla/tests:sample_text_test_gpu",
        "//tensorflow/compiler/xla/tests:scalar_computations_test_gpu",
        "//tensorflow/compiler/xla/tests:scatter_test_gpu",
        "//tensorflow/compiler/xla/tests:select_test_gpu",
        "//tensorflow/compiler/xla/tests:slice_test_gpu",
        "//tensorflow/compiler/xla/tests:transfer_manager_test_gpu",
        "//tensorflow/compiler/xla/tests:transpose_test_gpu",
        "//tensorflow/compiler/xla/tests:triangular_solve_test_gpu",
        "//tensorflow/compiler/xla/tests:tuple_test_gpu",
        "//tensorflow/compiler/xla/tests:unary_op_test_gpu",
        "//tensorflow/compiler/xla/tests:value_inference_test_gpu",
        "//tensorflow/compiler/xla/tests:vector_ops_reduce_test_gpu",
        "//tensorflow/compiler/xla/tests:vector_ops_simple_test_gpu",
        "//tensorflow/compiler/xla/tests:xla_hlo_profile_test_gpu",
    ] + if_google([
        # Currently fails in OSS.
        "//tensorflow/compiler/xla/tests:conv_depthwise_backprop_filter_test_gpu",
        "//tensorflow/compiler/xla/tests:conv_depthwise_test_gpu",
        "//tensorflow/compiler/xla/tests:convolution_dimension_numbers_test_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_1d_gpu_alternative_layout_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_1d_no_vmodule_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_gpu_alternative_layout_gpu",
        "//tensorflow/compiler/xla/tests:convolution_variants_test_gpu",
        "//tensorflow/compiler/xla/tests:grouped_convolution_test_gpu",
        "//tensorflow/python/kernel_tests/signal:fft_ops_test_xla_gpu",
    ]),
)

# These tests are intended to be run with --test_env=XLA_FLAGS=--xla_gpu_jitrt_executable
# See tap/tensorflow.xla_gpu_jitrt.
test_suite(
    name = "jitrt_executable_tests",
    tests = [
        "//tensorflow/compiler/tests:fft_test_gpu",
        "//tensorflow/compiler/xla/service/gpu:cudnn_fused_conv_rewriter_test",
        "//tensorflow/compiler/xla/service/gpu:custom_call_test",
        "//tensorflow/compiler/xla/service/gpu/tests:add_preds.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:all_reduce.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:bef_executable_test_gpu",
        "//tensorflow/compiler/xla/service/gpu/tests:concat.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:constant.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:copy.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:copy_nested.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:dynamic_update_slice_inplace.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:element_wise_row_vectorization.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:element_wise_row_vectorization_test",
        "//tensorflow/compiler/xla/service/gpu/tests:fused_scatter.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:fused_slice.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:fused_slice_different_operands.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:fusion.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:gemm_broadcast_folding_rewrite_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_alignment_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_atomic_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_copy_alone_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_copy_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_ftz_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_fusion_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_index_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_infeed_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_input_fusible_slice_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_kernel_tiling_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_ldg_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_noalias_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_reduce_scatter_creator_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_spmd_e2e_compile_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_too_many_blocks_test",
        "//tensorflow/compiler/xla/service/gpu/tests:gpu_unrolling_test",
        "//tensorflow/compiler/xla/service/gpu/tests:kernel_launch_test",
        "//tensorflow/compiler/xla/service/gpu/tests:launch_dimensions.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:mlir_fft_test",
        "//tensorflow/compiler/xla/service/gpu/tests:mlir_gemm_test",
        "//tensorflow/compiler/xla/service/gpu/tests:mlir_gpu_compile_test",
        "//tensorflow/compiler/xla/service/gpu/tests:pad_to_static.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:parallel_reduction_test",
        "//tensorflow/compiler/xla/service/gpu/tests:pred_arithmetic_test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduce_unnested.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduction_degenerate_dim_remover_test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduction_dimension_grouper_test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduction_layout_normalizer_test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduction_vectorization_sm_all.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:reduction_vectorization_test",
        "//tensorflow/compiler/xla/service/gpu/tests:rng_get_and_update_state.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:scatter.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:select_and_scatter.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:slice_to_dynamic.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:sorting.hlo.test",
        "//tensorflow/compiler/xla/service/gpu/tests:sorting_test",
        "//tensorflow/compiler/xla/service/gpu/tests:tree_reduction_rewriter_test",
        "//tensorflow/compiler/xla/tests:array_elementwise_ops_test_gpu",
        "//tensorflow/compiler/xla/tests:axpy_simple_test_gpu",
        "//tensorflow/compiler/xla/tests:bad_rng_shape_validation_test_gpu",
        "//tensorflow/compiler/xla/tests:batch_normalization_test_gpu",
        "//tensorflow/compiler/xla/tests:bfloat16_test_gpu",
        "//tensorflow/compiler/xla/tests:binop_scaling_test_gpu",
        "//tensorflow/compiler/xla/tests:bitcast_convert_test_gpu",
        "//tensorflow/compiler/xla/tests:broadcast_simple_test_gpu",
        "//tensorflow/compiler/xla/tests:broadcast_test_gpu",
        "//tensorflow/compiler/xla/tests:call_test_gpu",
        "//tensorflow/compiler/xla/tests:check_execution_arity_test_gpu",
        "//tensorflow/compiler/xla/tests:cholesky_test_gpu",
        "//tensorflow/compiler/xla/tests:client_test_gpu",
        "//tensorflow/compiler/xla/tests:compilation_cache_test_gpu",
        "//tensorflow/compiler/xla/tests:compute_constant_test_gpu",
        "//tensorflow/compiler/xla/tests:concat_test_gpu",
        "//tensorflow/compiler/xla/tests:constant_reduction_function_test_gpu",
        "//tensorflow/compiler/xla/tests:constants_test_gpu",
        "//tensorflow/compiler/xla/tests:conv_depthwise_backprop_filter_test_gpu",
        "//tensorflow/compiler/xla/tests:conv_depthwise_test_gpu",
        "//tensorflow/compiler/xla/tests:convert_test_gpu",
        "//tensorflow/compiler/xla/tests:convolution_dimension_numbers_test_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_1d_autotune_disabled_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_1d_gpu_alternative_layout_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_1d_no_vmodule_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_autotune_disabled_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_cudnn_frontend_disabled_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_gpu",
        "//tensorflow/compiler/xla/tests:convolution_test_gpu_alternative_layout_gpu",
        "//tensorflow/compiler/xla/tests:convolution_variants_test_gpu",
        "//tensorflow/compiler/xla/tests:copy_test_gpu",
        "//tensorflow/compiler/xla/tests:cpu_gpu_fusion_test_gpu",
        "//tensorflow/compiler/xla/tests:deallocation_test_gpu",
        "//tensorflow/compiler/xla/tests:deconstruct_tuple_test_gpu",
        "//tensorflow/compiler/xla/tests:deep_graph_test_gpu",
        "//tensorflow/compiler/xla/tests:dot_operation_single_threaded_runtime_test_gpu",
        "//tensorflow/compiler/xla/tests:dot_operation_test_gpu",
        "//tensorflow/compiler/xla/tests:dynamic_ops_test_gpu",
        "//tensorflow/compiler/xla/tests:execution_profile_test_gpu",
        "//tensorflow/compiler/xla/tests:execution_profile_test_with_xla_hlo_profile_gpu",
        "//tensorflow/compiler/xla/tests:exhaustive_binary_16_bit_test_gpu",
        "//tensorflow/compiler/xla/tests:exhaustive_binary_test_f32_f64_gpu",
        "//tensorflow/compiler/xla/tests:exhaustive_unary_test_complex_gpu",
        "//tensorflow/compiler/xla/tests:exhaustive_unary_test_f32_or_smaller_gpu",
        "//tensorflow/compiler/xla/tests:exhaustive_unary_test_f64_gpu",
        "//tensorflow/compiler/xla/tests:floor_ceil_test_gpu",
        "//tensorflow/compiler/xla/tests:fmax_fmin_test_gpu",
        "//tensorflow/compiler/xla/tests:gather_operation_test_gpu",
        "//tensorflow/compiler/xla/tests:get_dimension_size_test_gpu",
        "//tensorflow/compiler/xla/tests:grouped_convolution_test_gpu",
        "//tensorflow/compiler/xla/tests:half_test_gpu",
        "//tensorflow/compiler/xla/tests:iota_test_gpu",
        "//tensorflow/compiler/xla/tests:local_client_allocation_test_gpu",
        "//tensorflow/compiler/xla/tests:local_client_execute_test_gpu",
        "//tensorflow/compiler/xla/tests:log_test_gpu",
        "//tensorflow/compiler/xla/tests:map_test_gpu",
        "//tensorflow/compiler/xla/tests:matrix_ops_simple_test_gpu",
        "//tensorflow/compiler/xla/tests:multidimensional_slice_test_gpu",
        "//tensorflow/compiler/xla/tests:multioutput_fusion_test_gpu",
        "//tensorflow/compiler/xla/tests:pad_test_gpu",
        "//tensorflow/compiler/xla/tests:params_test_gpu",
        "//tensorflow/compiler/xla/tests:pred_test_gpu",
        "//tensorflow/compiler/xla/tests:prng_test_gpu",
        "//tensorflow/compiler/xla/tests:ptxas_bug_120501638_gpu",
        "//tensorflow/compiler/xla/tests:query_inferred_shape_test_gpu",
        "//tensorflow/compiler/xla/tests:reduce_hlo_test_gpu",
        "//tensorflow/compiler/xla/tests:reduce_precision_test_gpu",
        "//tensorflow/compiler/xla/tests:replay_test_gpu",
        "//tensorflow/compiler/xla/tests:reshape_motion_test_gpu",
        "//tensorflow/compiler/xla/tests:reshape_test_gpu",
        "//tensorflow/compiler/xla/tests:reverse_test_gpu",
        "//tensorflow/compiler/xla/tests:round_trip_packed_literal_test_gpu",
        "//tensorflow/compiler/xla/tests:round_trip_transfer_test_gpu",
        "//tensorflow/compiler/xla/tests:sample_text_test_gpu",
        "//tensorflow/compiler/xla/tests:scalar_computations_test_gpu",
        "//tensorflow/compiler/xla/tests:scatter_test_gpu",
        "//tensorflow/compiler/xla/tests:select_test_gpu",
        "//tensorflow/compiler/xla/tests:slice_test_gpu",
        "//tensorflow/compiler/xla/tests:transfer_manager_test_gpu",
        "//tensorflow/compiler/xla/tests:transpose_test_gpu",
        "//tensorflow/compiler/xla/tests:triangular_solve_test_gpu",
        "//tensorflow/compiler/xla/tests:tuple_test_gpu",
        "//tensorflow/compiler/xla/tests:unary_op_test_gpu",
        "//tensorflow/compiler/xla/tests:value_inference_test_gpu",
        "//tensorflow/compiler/xla/tests:vector_ops_reduce_test_gpu",
        "//tensorflow/compiler/xla/tests:vector_ops_simple_test_gpu",
        "//tensorflow/compiler/xla/tests:xla_hlo_profile_test_gpu",
    ] + if_google([
        # Currently fails in OSS.
        "//tensorflow/python/kernel_tests/signal:fft_ops_test_xla_gpu",
    ]),
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        "//tensorflow/core/lib/monitoring:sampler",
    ],
)

cc_library(
    name = "precompiled_kernels",
    srcs = if_gpu_is_configured(["precompiled_kernels.cc"]),
    hdrs = if_gpu_is_configured(["precompiled_kernels.h"]),
    deps = if_gpu_is_configured([
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:stream_header",
        "//tensorflow/stream_executor/gpu:asm_compiler",
        "//tensorflow/stream_executor/gpu:gpu_asm_opts",
    ]) + if_rocm_is_configured([
        "//tensorflow/stream_executor/gpu:gpu_stream_header",
    ]),
)

cc_library(
    name = "triangular_solve_rewriter",
    srcs = ["triangular_solve_rewriter.cc"],
    hdrs = ["triangular_solve_rewriter.h"],
    deps = [
        ":cublas_cudnn",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_creation_utils",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_library(
    name = "runtime_intrinsics",
    srcs = ["runtime_intrinsics.cc"],
    hdrs = ["runtime_intrinsics.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:custom_call_status",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/stream_executor",
        "@com_google_absl//absl/cleanup",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hlo_fusion_stats",
    srcs = ["hlo_fusion_stats.cc"],
    hdrs = ["hlo_fusion_stats.h"],
    deps = [
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "hlo_fusion_stats_test",
    srcs = ["hlo_fusion_stats_test.cc"],
    tags = ["no_pip"],
    deps = [
        ":hlo_fusion_stats",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)
