# Description:
#   Components that help with Triton autotuning.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/util:cc_embed_data.bzl", "cc_embed_data")

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
    name = "dot_search_space",
    srcs = ["dot_search_space.cc"],
    hdrs = ["dot_search_space.h"],
    tags = ["gpu"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla/backends/gpu/codegen/triton:tma_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service/gpu:matmul_utils",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/gpu:tma_metadata",
        "//xla/tsl/lib/core:bits",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf_lite",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "dot_search_space_test",
    srcs = ["dot_search_space_test.cc"],
    tags = ["gpu"],
    deps = [
        ":dot_search_space",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service/gpu:matmul_utils",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/rocm:rocm_compute_capability",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "all_default_configs",
    data = glob(["default_configs/*.txtpb"]),
)

exports_files(glob([
    "default_configs/*.txtpb",
]))

cc_embed_data(
    name = "embed_default_configs",
    srcs = glob(["default_configs/*.txtpb"]),
    outs = [
        "embed_default_configs.cc",
        "embed_default_configs.h",
    ],
    compatible_with = get_compatible_with_portable() + if_google(["//buildenv/target:libtpu"]),
    embedopts = ["--namespace=xla::gpu::configs"],
    flatten = True,
)

cc_library(
    name = "triton_configs",
    srcs = ["triton_configs.cc"],
    hdrs = ["triton_configs.h"],
    deps = [
        ":embed_default_configs",
        "//xla:autotuning_proto_cc",
        "//xla/service/gpu:matmul_utils",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
    ],
)

xla_cc_test(
    name = "triton_configs_test",
    srcs = ["triton_configs_test.cc"],
    deps = [
        ":triton_configs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cost_model_config_optimization",
    srcs = ["cost_model_config_optimization.cc"],
    hdrs = ["cost_model_config_optimization.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/transforms:convert_triton_gemm_config",
        "//xla/codegen/tiling:symbolic_tile_analysis",
        "//xla/codegen/tiling:tiled_hlo_computation",
        "//xla/codegen/tiling:tiling_specification",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:instruction_fusion",
        "//xla/service/gpu:matmul_utils",
        "//xla/service/gpu/model:block_level_parameters",
        "//xla/service/gpu/model:fusion_analysis_cache",
        "//xla/service/gpu/model:gpu_indexing_performance_model",
        "//xla/service/gpu/model:gpu_performance_model_base",
        "//xla/service/gpu/model:tiling_from_block_parameters",
        "//xla/service/gpu/model:triton_emitter_constraints",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:sorted_range",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "cost_model_config_optimization_test",
    srcs = ["cost_model_config_optimization_test.cc"],
    deps = [
        ":cost_model_config_optimization",
        "//xla:autotuning_proto_cc",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_query",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:matmul_utils",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf_lite",
        "@llvm-project//mlir:IR",
    ],
)
