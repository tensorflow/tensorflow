load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("//xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

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
    name = "nccl_errors",
    hdrs = if_gpu_is_configured(["nccl_errors.h"]),
    visibility = ["//visibility:private"],
    deps = [
        "//xla:util",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
    ],
)

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
