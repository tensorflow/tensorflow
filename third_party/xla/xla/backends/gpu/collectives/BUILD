load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google", "internal_visibility")
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

# Allows to explicitely disable nvshmem collectives using invocation flag.
bool_flag(
    name = "nvshmem_enabled",
    build_setting_default = True,
)

config_setting(
    name = "nvshmem_supported",
    constraint_values = [
        # TODO(b/409709288): Fix nvshmem ARM issues and remove this condition.
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    flag_values = {
        ":nvshmem_enabled": "True",
    },
)

# Since selects can't be nested we need to create this intermediate target
cc_library(
    name = "nvshmem_collectives_if_supported",
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps =
        select({
            ":nvshmem_supported": [":nvshmem_collectives"],
            "//conditions:default": [],
        }),
)

# Build target that registers all available GPU collectives implementations with the collectives
# registry at link time.
cc_library(
    name = "gpu_collectives_plugin",
    deps = [
        ":gpu_collectives_stub",
    ] + if_gpu_is_configured([
        ":nccl_collectives",
    ]) + if_cuda_is_configured([
        ":nvshmem_collectives_if_supported",
    ]),
)

cc_library(
    name = "gpu_clique",
    srcs = ["gpu_clique.cc"],
    hdrs = ["gpu_clique.h"],
    deps = [
        ":gpu_clique_key",
        "//xla/core/collectives:clique",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/service:lockable",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "gpu_clique_key",
    srcs = ["gpu_clique_key.cc"],
    hdrs = ["gpu_clique_key.h"],
    deps = [
        "//xla/core/collectives:clique_key",
        "//xla/service:global_device_id",
        "//xla/tsl/lib/gtl:int_type",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
    ],
)

xla_cc_test(
    name = "gpu_clique_key_test",
    srcs = ["gpu_clique_key_test.cc"],
    deps = [
        ":gpu_clique_key",
        "//xla/core/collectives:clique_id",
        "//xla/service:global_device_id",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/container:btree",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gpu_cliques",
    srcs = ["gpu_cliques.cc"],
    hdrs = ["gpu_cliques.h"],
    deps = [
        ":gpu_clique",
        ":gpu_clique_key",
        ":gpu_collectives",
        "//xla:debug_options_flags",
        "//xla:executable_run_options",
        "//xla:types",
        "//xla:util",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/service:global_device_id",
        "//xla/service:lockable",
        "//xla/service:rendezvous",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:hash",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "gpu_collectives",
    srcs = ["gpu_collectives.cc"],
    hdrs = ["gpu_collectives.h"],
    deps = [
        ":gpu_communicator",
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:collectives_registry",
        "//xla/core/collectives:communicator",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/service:global_device_id",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:casts",
    ],
)

cc_library(
    name = "gpu_communicator",
    hdrs = ["gpu_communicator.h"],
    deps = [
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/service:collective_ops_utils",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/concurrency:async_value",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
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
    local_defines =
        if_rocm_is_configured([
            "TENSORFLOW_USE_ROCM=1",
        ]),
    tags = ["gpu"],
    visibility = ["//visibility:private"],
    deps = [
        "//xla:util",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ] + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rccl",
    ]),
)

cc_library(
    name = "nccl_collectives",
    srcs = ["nccl_collectives.cc"],
    hdrs = ["nccl_collectives.h"],
    local_defines =
        if_rocm_is_configured([
            "TENSORFLOW_USE_ROCM=1",
        ]),
    tags = ["gpu"],
    visibility = ["//visibility:private"],
    deps = [
        ":gpu_clique_key",
        ":gpu_collectives",
        ":gpu_communicator",
        ":nccl_communicator",
        ":nccl_errors",
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:collectives_registry",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/service:collective_ops_utils",
        "//xla/service:global_device_id",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:numbers",
    ] + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rccl",
    ]),
    alwayslink = True,  # registers collectives implementation
)

cc_library(
    name = "nccl_communicator",
    srcs = ["nccl_communicator.cc"],
    hdrs = ["nccl_communicator.h"],
    local_defines = if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    tags = ["gpu"],
    visibility = ["//visibility:private"],
    deps = [
        ":gpu_collectives",
        ":gpu_communicator",
        ":nccl_errors",
        ":single_threaded_executor",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:collectives_registry",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/service:collective_ops_utils",
        "//xla/service:global_device_id",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:numbers",
    ] + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rccl",
    ]),
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
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:collectives_registry",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/service:collective_ops_utils",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:stream",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:numbers",
        "@nvshmem//:nvshmem_lib",
    ],
    alwayslink = True,
)

xla_test(
    name = "nccl_communicator_test",
    srcs = ["nccl_communicator_test.cc"],
    backends = ["gpu"],
    local_defines =
        if_rocm_is_configured([
            "TENSORFLOW_USE_ROCM=1",
        ]),
    tags = [
        # Stop chloroxylenol from running this test with msan because msan does
        # not work with CUDA.
        #
        # go/chloroxylenol-faq#how-do-i-stop-chloroxylenol-from-running-my-test
        # go/cuda#memorysanitizer
        "nomsan",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":gpu_collectives",
        ":nccl_collectives",
        ":nccl_communicator",
        ":nccl_errors",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/service:collective_ops_utils",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status_matchers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/utility",
        "@com_google_googletest//:gtest_main",
    ] + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rccl",
    ]),
)

xla_cc_test(
    name = "nvshmem_collectives_test",
    srcs = ["nvshmem_collectives_test.cc"],
    tags = [
        "cuda-only",
        "gpu",
        "no_oss",
        "nomsan",
        "notap",  # TODO(b/399931591): Re-enable once flakiness is resolved.
    ] + if_google(google_value = ["requires-gpu-nvidia:2"]),  # This tag is not documented in OSS yet.
    deps = [
        ":nvshmem_collectives",
        "//xla:debug_options_flags",
        "//xla:status_macros",
        "//xla/pjrt/distributed",
        "//xla/pjrt/distributed:client",
        "//xla/pjrt/distributed:service",
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
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:threadpool_async_executor",
    ],
)

xla_cc_test(
    name = "single_threaded_executor_test",
    srcs = ["single_threaded_executor_test.cc"],
    deps = [
        ":single_threaded_executor",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)
