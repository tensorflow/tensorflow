load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/stream_executor:build_defs.bzl", "if_cuda_or_rocm_is_configured")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

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

alias(
    # Since RCCL reimplements the NCCL API and there is no way to disable that,
    # we need to make sure that we only link either NCCL or RCCL.
    name = "nccl_or_rccl_collectives",
    actual = if_rocm_is_configured(
        ":rccl_collectives",
        ":nccl_collectives",
    ),
    tags = ["manual"],
)

# Build target that registers all available GPU collectives implementations with the collectives
# registry at link time.
cc_library(
    name = "gpu_collectives_plugin",
    deps = [
        ":gpu_collectives_stub",
    ] + if_cuda_or_rocm_is_configured([
        ":nccl_or_rccl_collectives",
    ]) + if_cuda_is_configured([
        ":nvshmem_collectives_if_supported",
    ]),
)

cc_library(
    name = "cancellation_token",
    hdrs = ["cancellation_token.h"],
)

cc_library(
    name = "gpu_clique",
    srcs = ["gpu_clique.cc"],
    hdrs = ["gpu_clique.h"],
    deps = [
        ":cancellation_token",
        ":gpu_clique_key",
        ":gpu_communicator",
        "//xla:util",
        "//xla/core/collectives:clique",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/service:lockable",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "gpu_clique_key",
    srcs = ["gpu_clique_key.cc"],
    hdrs = ["gpu_clique_key.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives:clique_key",
        "//xla/runtime:device_id",
        "//xla/tsl/lib/gtl:int_type",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:casts",
    ],
)

xla_cc_test(
    name = "gpu_clique_key_test",
    srcs = ["gpu_clique_key_test.cc"],
    deps = [
        ":gpu_clique_key",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives:clique_id",
        "//xla/runtime:device_id",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/container:btree",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gpu_clique_rendezvous",
    srcs = ["gpu_clique_rendezvous.cc"],
    hdrs = ["gpu_clique_rendezvous.h"],
    deps = [
        ":gpu_clique_key",
        "//xla:util",
        "//xla/core/collectives:rank_id",
        "//xla/service:rendezvous",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "gpu_clique_rendezvous_test",
    srcs = ["gpu_clique_rendezvous_test.cc"],
    deps = [
        ":gpu_clique_key",
        ":gpu_clique_rendezvous",
        "//xla/core/collectives:rank_id",
        "//xla/runtime:device_id",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gpu_cliques",
    srcs = ["gpu_cliques.cc"],
    hdrs = ["gpu_cliques.h"],
    deps = [
        ":cancellation_token",
        ":gpu_clique",
        ":gpu_clique_key",
        ":gpu_collectives",
        "//xla:debug_options_flags",
        "//xla:executable_run_options",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/runtime:device_id",
        "//xla/service:lockable",
        "//xla/service:rendezvous",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:hash",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "gpu_collectives",
    srcs = ["gpu_collectives.cc"],
    hdrs = ["gpu_collectives.h"],
    deps = [
        ":cancellation_token",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:collectives_registry",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/runtime:device_id",
        "//xla/runtime:process_id",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:casts",
    ],
)

xla_cc_test(
    name = "gpu_collectives_test",
    srcs = ["gpu_collectives_test.cc"],
    tags = [
        "cuda-only",
        "gpu",
        "requires-gpu-nvidia:2",
    ],
    deps = [
        ":cancellation_token",
        ":gpu_clique_key",
        ":gpu_collectives",
        ":gpu_communicator",
        ":nccl_or_rccl_collectives",
        "//xla:future",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:rank_id",
        "//xla/core/collectives:reduction_kind",
        "//xla/core/collectives:symmetric_memory",
        "//xla/runtime:device_id",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_allocator",
        "//xla/stream_executor:memory_space",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_platform",
        "//xla/tsl/concurrency:executor",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_communicator",
    hdrs = ["gpu_communicator.h"],
    deps = [
        "//xla:future",
        "//xla:util",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/core/collectives:symmetric_memory",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel_args",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gpu_collectives_stub",
    srcs = ["gpu_collectives_stub.cc"],
    hdrs = ["gpu_collectives_stub.h"],
    deps = [
        ":gpu_collectives",
        "//xla:util",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:collectives_registry",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = True,  # registers collectives implementation
)

cc_library(
    name = "nccl_errors",
    srcs = ["nccl_errors.cc"],
    hdrs = ["nccl_errors.h"],
    tags = [
        "cuda-only",
        "gpu",
        "no-oneapi",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":cancellation_token",
        "//xla:util",
        "//xla/tsl/cuda:nccl",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "nccl_symmetric_memory",
    srcs = ["nccl_symmetric_memory.cc"],
    hdrs = ["nccl_symmetric_memory.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":nccl_errors",
        "//xla/core/collectives:symmetric_memory",
        "//xla/stream_executor:device_address",
        "//xla/tsl/cuda:nccl",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "rccl_errors",
    srcs = ["rccl_errors.cc"],
    hdrs = ["rccl_errors.h"],
    tags = [
        "gpu",
        "no-oneapi",
        "rocm-only",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":cancellation_token",
        "//xla:util",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@local_config_rocm//rocm:rccl",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

cc_library(
    name = "nccl_collectives",
    srcs = ["nccl_collectives.cc"],
    hdrs = ["nccl_collectives.h"],
    tags = [
        "cuda-only",
        "gpu",
        "no-oneapi",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":cancellation_token",
        ":gpu_clique_key",
        ":gpu_collectives",
        ":nccl_communicator",
        ":nccl_errors",
        "//xla:debug_options_flags",
        "//xla:status_macros",
        "//xla:util",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:collectives_registry",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/runtime:device_id",
        "//xla/runtime:process_id",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:nccl_memory_allocator",  # buildcleaner: keep (static registration)
        "//xla/tsl/cuda:nccl",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:numbers",
    ],
    alwayslink = True,  # registers collectives implementation
)

cc_library(
    name = "rccl_collectives",
    srcs = ["rccl_collectives.cc"],
    hdrs = ["rccl_collectives.h"],
    tags = [
        "gpu",
        "no-oneapi",
        "rocm-only",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":cancellation_token",
        ":gpu_clique_key",
        ":gpu_collectives",
        ":rccl_communicator",
        ":rccl_errors",
        "//xla:debug_options_flags",
        "//xla:status_macros",
        "//xla:util",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:collectives_registry",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/runtime:device_id",
        "//xla/runtime:process_id",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_config_rocm//rocm:rccl",  # buildcleaner: keep
        "@local_config_rocm//rocm:rocm_headers",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:numbers",
    ],
    alwayslink = True,  # registers collectives implementation
)

cc_library(
    name = "nccl_communicator",
    srcs = ["nccl_communicator.cc"],
    hdrs = ["nccl_communicator.h"],
    tags = [
        "cuda-only",
        "gpu",
        "no-oneapi",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":cancellation_token",
        ":gpu_collectives",
        ":gpu_communicator",
        ":nccl_errors",
        ":nccl_symmetric_memory",
        ":single_threaded_executor",
        "//xla:future",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/core/collectives:reduction_kind",
        "//xla/core/collectives:symmetric_memory",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel_args",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/concurrency:executor",
        "//xla/tsl/cuda:nccl",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/platform:casts",
    ],
)

cc_library(
    name = "rccl_communicator",
    srcs = ["rccl_communicator.cc"],
    hdrs = ["rccl_communicator.h"],
    tags = [
        "gpu",
        "no-oneapi",
        "rocm-only",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":cancellation_token",
        ":gpu_collectives",
        ":gpu_communicator",
        ":rccl_errors",
        ":single_threaded_executor",
        "//xla:future",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/core/collectives:reduction_kind",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/concurrency:executor",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_rocm//rocm:rccl",  # buildcleaner: keep
        "@local_config_rocm//rocm:rocm_headers",
        "@tsl//tsl/platform:casts",
    ],
)

cc_library(
    name = "nvshmem_collectives",
    srcs = [
        "nvshmem_collectives.cc",
        "nvshmem_communicator.cc",
    ],
    hdrs = [
        "nvshmem_collectives.h",
        "nvshmem_communicator.h",
    ],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":gpu_collectives",
        "//xla:future",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:collectives_registry",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/core/collectives:reduction_kind",
        "//xla/service:collective_ops_utils",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:stream",
        "//xla/stream_executor/cuda:nvshmem",
        "//xla/stream_executor/cuda:nvshmem_memory_allocator",  # buildcleaner: keep (static registration)
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_config_cuda//cuda:cuda_headers",
        "@nvshmem//:nvshmem_lib",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:numbers",
    ],
    alwayslink = True,
)

cc_library(
    name = "nvshmem_collectives_if_builtin_used",
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = select({
        "//xla/stream_executor/cuda:no_builtin_used": [],
        "//conditions:default": [":nvshmem_collectives"],
    }),
)

cc_library(
    name = "nvshmem_collectives_if_supported",
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = select({
        "//xla/stream_executor/cuda:nvshmem_supported": [
            ":nvshmem_collectives_if_builtin_used",
        ],
        "//conditions:default": [],
    }),
)

xla_test(
    name = "nvshmem_collectives_test",
    srcs = ["nvshmem_collectives_test.cc"],
    backend_tags = {
        "gpu": [
            "multi_gpu",
            "no_oss",
            "noasan",
            "nomsan",
            "requires-gpu-nvidia",
        ],
        "b200": [
            "multi_gpu",
            "broken",
        ],
        "nvgpu_any": [
            "broken",
            "no_oss",
        ],
    },
    backends = ["gpu"],
    env = {
        "XLA_FLAGS": "--xla_gpu_experimental_enable_nvshmem=true",
    },
    tags = ["cuda-only"],
    deps = [
        ":nvshmem_collectives",
        "//xla:debug_options_flags",
        "//xla:status_macros",
        "//xla/core/collectives:communicator",
        "//xla/pjrt/distributed",
        "//xla/pjrt/distributed:client",
        "//xla/pjrt/distributed:service",
        "//xla/runtime:process_id",
        "//xla/stream_executor/cuda:nvshmem",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:subprocess",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "single_threaded_executor",
    srcs = ["single_threaded_executor.cc"],
    hdrs = ["single_threaded_executor.h"],
    deps = [
        "//xla/tsl/concurrency:executor",
        "//xla/tsl/platform:env",
    ],
)

xla_cc_test(
    name = "single_threaded_executor_test",
    srcs = ["single_threaded_executor_test.cc"],
    deps = [
        ":single_threaded_executor",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)
