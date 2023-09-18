load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_tsl//tsl:tsl.bzl", "if_nccl")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")
load("@local_tsl//tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")
load("@local_tsl//tsl/platform:build_config.bzl", "tf_proto_library")
load("@local_tsl//tsl/platform:build_config_root.bzl", "tf_cuda_tests_tags")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "gpu_helpers",
    srcs = ["gpu_helpers.cc"],
    hdrs = ["gpu_helpers.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/client:client_library",
        "//xla/client:local_client",
        "//xla/service:platform_util",
        "//xla/stream_executor:device_mem_allocator",
        "//xla/stream_executor:kernel",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/framework:bfc_allocator",
        "@local_tsl//tsl/framework:device_id_impl",
        "@local_tsl//tsl/util:env_var",
    ],
)

cc_library(
    name = "se_gpu_pjrt_client",
    srcs = ["se_gpu_pjrt_client.cc"],
    hdrs = ["se_gpu_pjrt_client.h"],
    defines = if_cuda(["GOOGLE_CUDA=1"]) + if_rocm(["TENSORFLOW_USE_ROCM=1"]),
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_helpers",
        ":gpu_topology",
        "//xla:statusor",
        "//xla:util",
        "//xla/client:client_library",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/pjrt:tracked_device_buffer",
        "//xla/pjrt:utils",
        "//xla/pjrt/distributed:client",
        "//xla/pjrt/distributed:topology_util",
        "//xla/service:platform_util",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/stream_executor:device_mem_allocator",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:tf_allocator_adapter",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/framework:bfc_allocator",
        "@local_tsl//tsl/framework:device_id",
        "@local_tsl//tsl/framework:device_id_impl",
        "@local_tsl//tsl/lib/strings:proto_serialization",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/profiler/lib:connected_traceme",
        "@local_tsl//tsl/util:env_var",
        "@tf_runtime//:hostcontext",
    ] + if_cuda([
        ":nccl_id_store_cuda",
        "@local_config_cuda//cuda:cuda_headers",
        "//xla/stream_executor/cuda:cuda_activation_header",
        "//xla/stream_executor/gpu:gpu_cudamallocasync_allocator",
    ]) + if_rocm([
        ":nccl_id_store_rocm",
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

xla_cc_test(
    name = "se_gpu_pjrt_client_test",
    srcs = if_gpu_is_configured(["se_gpu_pjrt_client_test.cc"]),
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    tags = [
        "gpu",
        "no_oss",
        "noasan",
        "nomsan",
        "requires-gpu-nvidia:2",
    ],
    deps = [
        ":se_gpu_pjrt_client",
        "//xla:statusor",
        "//xla:test",
        "//xla/pjrt:utils",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_parser",
        "//xla/tests:literal_test_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test_main",
        "@tf_runtime//:hostcontext",
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
    visibility = ["//visibility:public"],
    deps = [
        "//xla:statusor",
        "//xla:util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/distributed:client",
        "//xla/service:global_device_id",
        "//xla/service/gpu:gpu_executable_run_options",
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
    visibility = ["//visibility:public"],
    deps = [
        "//xla:statusor",
        "//xla:util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/distributed:client",
        "//xla/service:global_device_id",
        "//xla/service/gpu:gpu_executable_run_options",
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
        "//xla/pjrt:pjrt_client_test_common",
        "//xla/service:gpu_plugin",
        "@local_tsl//tsl/platform:test_main",
    ],
)

tf_proto_library(
    name = "gpu_topology_proto",
    srcs = ["gpu_topology.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gpu_topology",
    srcs = ["gpu_topology.cc"],
    hdrs = ["gpu_topology.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_topology_proto_cc",
    ],
)

cc_library(
    name = "se_gpu_pjrt_compiler",
    srcs = ["se_gpu_pjrt_compiler.cc"],
    hdrs = ["se_gpu_pjrt_compiler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":se_gpu_pjrt_client",
        "//xla:status_macros",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "@com_google_absl//absl/status",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "se_gpu_pjrt_compiler_test",
    srcs = if_gpu_is_configured(["se_gpu_pjrt_compiler_test.cc"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":se_gpu_pjrt_client",
        ":se_gpu_pjrt_compiler",
        "//xla/client:xla_computation",
        "//xla/mlir_hlo",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_parser",
        "//xla/stream_executor/cuda:cublas_plugin",
        "//xla/tests:literal_test_util",
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Parser",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test_main",
    ],
)
