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
        "@com_google_absl//absl/container:btree",
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
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "gpu_clique_key_test",
    srcs = ["gpu_clique_key_test.cc"],
    deps = [
        ":gpu_clique_key",
        "//xla/core/collectives:clique_id",
        "//xla/service:global_device_id",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "gpu_collectives",
    hdrs = ["gpu_collectives.h"],
    deps = [
        "//xla/core/collectives",
    ],
)

cc_library(
    name = "nccl_errors",
    hdrs = if_gpu_is_configured(["nccl_errors.h"]),
    visibility = ["//visibility:private"],
    deps = [
        "//xla:util",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
    ],
)

# TODO(b/380457503): Update visibility to "//visibility:private".
cc_library(
    name = "nccl_collectives",
    hdrs = if_gpu_is_configured(["nccl_collectives.h"]),
    local_defines = if_cuda_is_configured([
        "GOOGLE_CUDA=1",
    ]) + if_rocm_is_configured([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    deps = [
        ":nccl_communicator",
        "//xla/core/collectives",
    ] + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rccl",
    ]),
)

# TODO(b/380457503): Update visibility to "//visibility:private".
cc_library(
    name = "nccl_clique",
    srcs = ["nccl_clique.cc"],
    hdrs = ["nccl_clique.h"],
    deps = [
        ":gpu_clique",
        "//xla/core/collectives:clique",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/service/gpu/runtime:nccl_api",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
    ],
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
        ":nccl_errors",
        "//xla/core/collectives:communicator",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
    ] + if_cuda_is_configured([
        "@local_config_nccl//:nccl",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rccl",
    ]),
)
