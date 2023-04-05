load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl:tsl.bzl", "if_nccl")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_test")
load("//tensorflow/compiler/xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")
load("//tensorflow/tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/xla:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "gpu_helpers",
    srcs = ["gpu_helpers.cc"],
    hdrs = ["gpu_helpers.h"],
    visibility = ["//tensorflow/compiler/xla/pjrt:friends"],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/stream_executor:device_mem_allocator",
        "//tensorflow/compiler/xla/stream_executor:kernel",
        "//tensorflow/tsl/framework:bfc_allocator",
        "//tensorflow/tsl/framework:device_id_impl",
        "//tensorflow/tsl/util:env_var",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "se_gpu_pjrt_client",
    srcs = ["se_gpu_pjrt_client.cc"],
    hdrs = ["se_gpu_pjrt_client.h"],
    defines = if_cuda(["GOOGLE_CUDA=1"]) + if_rocm(["TENSORFLOW_USE_ROCM=1"]),
    visibility = ["//tensorflow/compiler/xla/pjrt:friends"],
    deps = [
        ":gpu_helpers",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/pjrt:pjrt_stream_executor_client",
        "//tensorflow/compiler/xla/pjrt/distributed:client",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
        "//tensorflow/compiler/xla/stream_executor:device_mem_allocator",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "//tensorflow/compiler/xla/stream_executor:tf_allocator_adapter",
        "//tensorflow/tsl/framework:bfc_allocator",
        "//tensorflow/tsl/framework:device_id",
        "//tensorflow/tsl/framework:device_id_impl",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/util:env_var",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ] + if_cuda([
        ":nccl_id_store_cuda",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_activation_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_cudamallocasync_allocator",
    ]) + if_rocm([
        ":nccl_id_store_rocm",
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

xla_cc_test(
    name = "se_gpu_pjrt_client_test",
    srcs = if_gpu_is_configured(["se_gpu_pjrt_client_test.cc"]),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":se_gpu_pjrt_client",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
)

# We actually wish we could write if_cuda(if_nccl(...)) in :gpu_device,
# but Bazel does not allow nested selects. We can work around the problem using
# an intermediate library that has the conditional NCCL pieces that is only
# itself included as a dependency if CUDA is enabled.
cc_library(
    name = "nccl_id_store_cuda",
    srcs = ["nccl_id_store.cc"],
    hdrs = ["nccl_id_store.h"],
    defines = if_nccl(["NCCL_ENABLED=1"]),
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/pjrt/distributed:client",
        "//tensorflow/compiler/xla/service:global_device_id",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ] + if_nccl(["@local_config_nccl//:nccl"]),
)

cc_library(
    name = "nccl_id_store_rocm",
    srcs = ["nccl_id_store.cc"],
    hdrs = ["nccl_id_store.h"],
    defines = if_nccl(["NCCL_ENABLED=1"]),
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/pjrt/distributed:client",
        "//tensorflow/compiler/xla/service:global_device_id",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ] + if_nccl(["@local_config_nccl//:nccl"]),
)

xla_cc_test(
    name = "pjrt_client_test_se_gpu",
    srcs = ["pjrt_client_test_se_gpu.cc"],
    tags = [
        "no_oss",
        "notap",
        "requires-gpu-nvidia:2",
    ],
    deps = [
        ":se_gpu_pjrt_client",
        "//tensorflow/compiler/xla/pjrt:pjrt_client_test_common",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/tsl/platform:test_main",
    ],
)
