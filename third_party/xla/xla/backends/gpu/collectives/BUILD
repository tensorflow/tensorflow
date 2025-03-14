load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")
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

# Build target that registers all available GPU collectives implementations with the collectives
# registry at link time.
cc_library(
    name = "gpu_collectives_plugin",
    deps = [
        ":gpu_collectives_stub",
        ":nccl_collectives",
    ],
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
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
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
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
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
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:collectives_registry",
        "//xla/core/collectives:communicator",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:casts",
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
    hdrs = if_gpu_is_configured(["nccl_errors.h"]),
    visibility = ["//visibility:private"],
    deps = [
        "//xla:util",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/strings:str_format",
    ],
)

# TODO(b/380457503): Update visibility to "//visibility:private".
cc_library(
    name = "nccl_collectives",
    srcs = if_gpu_is_configured(["nccl_collectives.cc"]),
    hdrs = if_gpu_is_configured(["nccl_collectives.h"]),
    local_defines = if_cuda_is_configured([
        "GOOGLE_CUDA=1",
    ]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        ":gpu_collectives",
        ":nccl_communicator",
        ":nccl_errors",
        "//xla:status_macros",
        "//xla:util",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:collectives_registry",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
    ] + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rccl",
    ]),
    alwayslink = True,  # registers collectives implementation
)

# TODO(b/380457503): Update visibility to "//visibility:private".
cc_library(
    name = "nccl_communicator",
    srcs = if_gpu_is_configured(["nccl_communicator.cc"]),
    hdrs = if_gpu_is_configured(["nccl_communicator.h"]),
    local_defines = if_cuda_is_configured([
        "GOOGLE_CUDA=1",
    ]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        ":gpu_collectives",
        ":nccl_errors",
        "//xla:shape_util",
        "//xla:util",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/service:collective_ops_utils",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_stream",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
    ] + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rccl",
    ]),
)

cc_library(
    name = "nvshmem_collectives",
    srcs = ["nvshmem_collectives.cc"],
    hdrs = ["nvshmem_collectives.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:collectives_registry",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:numbers",
        "@nvshmem//:nvshmem_lib",
    ],
    alwayslink = True,  # registers collectives implementation
)

xla_cc_test(
    name = "nccl_communicator_test",
    srcs = if_gpu_is_configured(["nccl_communicator_test.cc"]),
    local_defines = if_cuda_is_configured([
        "GOOGLE_CUDA=1",
    ]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    tags = [
        "requires-gpu-nvidia",
        # Stop chloroxylenol from running this test with msan because msan does
        # not work with CUDA.
        #
        # go/chloroxylenol-faq#how-do-i-stop-chloroxylenol-from-running-my-test
        # go/cuda#memorysanitizer
        "nomsan",
    ],
    deps = [
        ":gpu_collectives",
        ":nccl_communicator",
        ":nccl_errors",
        "//xla/core/collectives:rank_id",
        "//xla/service:collective_ops_utils",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
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
