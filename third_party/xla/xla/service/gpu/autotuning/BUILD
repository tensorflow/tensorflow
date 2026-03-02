# Description:
#   Components that implement GPU autotuning.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load("//xla/util:build_defs.bzl", "embed_files")

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
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "autotuner_pass_test",
    srcs = ["autotuner_pass_test.cc"],
    backends = ["gpu"],
    tags = ["cuda-only"],
    deps = [
        ":autotuner_pass",
        "//xla/backends/autotuner",
        "//xla/backends/autotuner:autotuner_cache_proto_cc",
        "//xla/backends/autotuner:codegen_backend",
        "//xla/backends/autotuner:profiler",
        "//xla/backends/gpu/autotuner:cublas",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:platform_util",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/gpu:nvptx_compiler_impl",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "autotuner_status_key",
    srcs = ["autotuner_status_key.cc"],
    hdrs = ["autotuner_status_key.h"],
    compatible_with = get_compatible_with_portable(),
    deps = ["@com_google_absl//absl/strings"],
)

cc_library(
    name = "autotune_cache_key",
    srcs = ["autotune_cache_key.cc"],
    hdrs = ["autotune_cache_key.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_cc_test(
    name = "autotune_cache_key_test",
    srcs = ["autotune_cache_key_test.cc"],
    data = ["//xla/backends/gpu/target_config:all_gpu_specs"],
    deps = [
        ":autotune_cache_key",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "autotuner_util",
    srcs = ["autotuner_util.cc"],
    hdrs = ["autotuner_util.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":autotune_cache_key",
        ":autotuner_status_key",
        "//xla:autotune_results_proto_cc",
        "//xla:autotuning_proto_cc",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:dump",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:base64",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:protobuf",
    ],
)

# We need a separate target, as runtime executable cannot depend on compilation
# pipeline.
cc_library(
    name = "autotuner_compile_util",
    srcs = ["autotuner_compile_util.cc"],
    hdrs = ["autotuner_compile_util.h"],
    tags = ["gpu"],
    deps = [
        ":autotuner_util",
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/autotuner:gpu_codegen_backend",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/service:maybe_owning_device_address",
        "//xla/service:shaped_buffer",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:kernel_stats",
        "//xla/stream_executor:stream",
        "//xla/stream_executor/gpu:redzone_allocator",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "redzone_buffers",
    srcs = ["redzone_buffers.cc"],
    hdrs = ["redzone_buffers.h"],
    deps = [
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:stream",
        "//xla/stream_executor/gpu:redzone_allocator",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

xla_test(
    name = "redzone_buffers_test",
    srcs = ["redzone_buffers_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":redzone_buffers",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:platform_util",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests:hlo_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "gpu_autotuning_proto",
    srcs = ["gpu_autotuning.proto"],
    deps = [
        "//xla:autotuning_proto",
        "//xla:xla_data_proto",
        "//xla/service:hlo_proto",
        "//xla/service/gpu:backend_configs",
    ],
)

xla_cc_test(
    name = "autotuner_util_test",
    srcs = ["autotuner_util_test.cc"],
    data = ["//xla/backends/gpu/target_config:all_gpu_specs"],
    tags = [
        "gpu",
        "pjrt_migration_candidate",
    ],
    deps = [
        ":autotune_cache_key",
        ":autotuner_status_key",
        ":autotuner_util",
        "//xla:autotune_results_proto_cc",
        "//xla:autotuning_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:dump",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor/host:host_platform",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "autotuner_pass",
    srcs = [
        "autotuner_pass.cc",
    ],
    hdrs = ["autotuner_pass.h"],
    tags = ["gpu"],
    deps = [
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/backends/autotuner",
        "//xla/backends/autotuner:autotuner_cache_interface",
        "//xla/backends/autotuner:codegen_backend",
        "//xla/backends/autotuner:profiler",
        "//xla/backends/gpu/autotuner:gpu_profiler",
        "//xla/backends/gpu/autotuner:legacy_cache",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/service:compiler",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

filegroup(
    name = "all_default_configs",
    data = glob(["default_configs/*.txtpb"]),
)

exports_files(glob([
    "default_configs/*.txtpb",
]))

embed_files(
    name = "embed_default_configs",
    srcs = glob(["default_configs/*.txtpb"]),
    compatible_with = get_compatible_with_portable() + if_google(["//buildenv/target:libtpu"]),
    cpp_namespace = "xla::gpu::configs",
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
