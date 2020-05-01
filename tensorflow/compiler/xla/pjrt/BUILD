load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "worker_thread",
    srcs = ["worker_thread.cc"],
    hdrs = ["worker_thread.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "event_pool",
    srcs = ["event_pool.cc"],
    hdrs = ["event_pool.h"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "semaphore",
    srcs = ["semaphore.cc"],
    hdrs = ["semaphore.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "semaphore_test",
    srcs = ["semaphore_test.cc"],
    deps = [
        ":semaphore",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "tracked_device_buffer",
    srcs = ["tracked_device_buffer.cc"],
    hdrs = ["tracked_device_buffer.h"],
    deps = [
        ":event_pool",
        ":local_device_state",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:device_memory_allocator",
        "//tensorflow/stream_executor:event",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "tracked_device_buffer_test",
    srcs = ["tracked_device_buffer_test.cc"],
    deps = [
        ":tracked_device_buffer",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:device_memory_allocator",
    ],
)

cc_library(
    name = "local_device_state",
    srcs = ["local_device_state.cc"],
    hdrs = ["local_device_state.h"],
    deps = [
        ":event_pool",
        ":semaphore",
        ":worker_thread",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor",
        "//tensorflow/stream_executor:event",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "pjrt_client",
    srcs = ["pjrt_client.cc"],
    hdrs = ["pjrt_client.h"],
    visibility = ["//tensorflow/compiler/xla:friends"],
    deps = [
        ":event_pool",
        ":local_device_state",
        ":tracked_device_buffer",
        "//tensorflow/compiler/xla:cpu_function_runtime",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/pjrt/distributed:protocol_proto_cc",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
        "//tensorflow/core:allocator",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:stream",
        "//tensorflow/stream_executor/host:host_platform_id",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "cpu_device",
    srcs = ["cpu_device.cc"],
    hdrs = ["cpu_device.h"],
    deps = [
        ":pjrt_client",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/service:platform_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "nvidia_gpu_device",
    srcs = ["nvidia_gpu_device.cc"],
    hdrs = ["nvidia_gpu_device.h"],
    copts = if_cuda(["-DNCCL_ENABLED=1"]),
    deps = [
        ":pjrt_client",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/pjrt/distributed:client",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core/common_runtime:bfc_allocator",
        "//tensorflow/core/common_runtime/gpu:gpu_mem_allocator",
        "//tensorflow/stream_executor:tf_allocator_adapter",
    ] + if_cuda(["@local_config_nccl//:nccl"]),
)

tf_cc_test(
    name = "gpu_multistream_test",
    srcs = ["gpu_multistream_test.cc"],
    tags = [
        # TODO(phawkins): figure out TF test infra such that this only runs under GPU.
        "no_oss",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":nvidia_gpu_device",
        ":pjrt_client",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:random",
    ],
)
