load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "if_nccl")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

package(
    default_visibility = ["//tensorflow:internal"],
    features = [
        "-layering_check",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow:internal",
    ],
    packages = [
        "//third_party/py/jax/...",
    ],
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
        "//tensorflow/core/platform:stream_executor",
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
    visibility = [
        "//learning/pathways/data_parallel:__pkg__",
        "//tensorflow:internal",
    ],
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
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
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
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//tensorflow/compiler/xla:friends"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        "//tensorflow/core/lib/monitoring:counter",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "pjrt_stream_executor_client",
    srcs = ["pjrt_stream_executor_client.cc"],
    hdrs = ["pjrt_stream_executor_client.h"],
    visibility = ["//tensorflow/compiler/xla:friends"],
    deps = [
        ":event_pool",
        ":local_device_state",
        ":metrics",
        ":pjrt_client",
        ":tracked_device_buffer",
        ":utils",
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
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:stream",
        "//tensorflow/stream_executor/host:host_platform_id",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tpu_client",
    srcs = ["tpu_client.cc"],
    hdrs = ["tpu_client.h"],
    visibility = [
        "//learning/brain/research/jax:__subpackages__",
        "//learning/deepmind/tensorflow/tensorfn:__subpackages__",
        "//learning/pathways:__subpackages__",
        "//tensorflow/compiler/xla:friends",
    ],
    deps = [
        ":local_device_state",
        ":pjrt_stream_executor_client",
        ":tracked_device_buffer",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:tpu_computation_placer",
        "//tensorflow/core:lib",
        "//tensorflow/core/tpu:tpu_on_demand_compiler",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:stream",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/stream_executor/tpu:tpu_executable",
        "//tensorflow/stream_executor/tpu:tpu_executable_interface",
        "//tensorflow/stream_executor/tpu:tpu_executor",
        "//tensorflow/stream_executor/tpu:tpu_executor_interface",
        "//tensorflow/stream_executor/tpu:tpu_platform_interface",
        "//tensorflow/stream_executor/tpu:tpu_topology_external",
        "//tensorflow/stream_executor/tpu:tpu_transfer_manager",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "interpreter_device",
    srcs = ["interpreter_device.cc"],
    hdrs = ["interpreter_device.h"],
    deps = [
        ":pjrt_stream_executor_client",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/service:interpreter_plugin",
        "//tensorflow/compiler/xla/service:platform_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cpu_device",
    srcs = ["cpu_device.cc"],
    hdrs = ["cpu_device.h"],
    visibility = [":friends"],
    deps = [
        ":pjrt_stream_executor_client",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:platform_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_device",
    srcs = ["gpu_device.cc"],
    hdrs = ["gpu_device.h"],
    defines = if_cuda(["GOOGLE_CUDA=1"]),
    deps = [
        ":pjrt_stream_executor_client",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/pjrt/distributed:client",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core/common_runtime:bfc_allocator",
        "//tensorflow/core/common_runtime/device:device_mem_allocator",
        "//tensorflow/core/common_runtime/device:device_id",
        "//tensorflow/core:lib_internal",
        "//tensorflow/stream_executor/cuda:cuda_activation_header",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:tf_allocator_adapter",
    ] + if_nccl([":nccl_plugin"]),
)

# We actually wish we could write if_cuda(if_nccl(...)) in :gpu_device,
# but Bazel does not allow nested selects. We can work around the problem using
# an intermediate library.
cc_library(
    name = "nccl_plugin",
    defines = if_cuda(["NCCL_ENABLED=1"]),
    deps = if_cuda(["@local_config_nccl//:nccl"]),
)

tf_cc_test(
    name = "gpu_multistream_test",
    srcs = ["gpu_multistream_test.cc"],
    tags = [
        # TODO(phawkins): figure out TF test infra such that this only runs under GPU.
        "no_oss",
        "requires-gpu-nvidia",
        "notap",
    ],
    deps = [
        ":gpu_device",
        ":pjrt_client",
        ":pjrt_stream_executor_client",
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

cc_library(
    name = "tracked_tfrt_cpu_device_buffer",
    srcs = ["tracked_tfrt_cpu_device_buffer.cc"],
    hdrs = ["tracked_tfrt_cpu_device_buffer.h"],
    deps = [
        "//tensorflow/compiler/xla:cpu_function_runtime",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/core/platform:platform_port",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "tfrt_cpu_pjrt_client",
    srcs = ["tfrt_cpu_pjrt_client.cc"],
    hdrs = ["tfrt_cpu_pjrt_client.h"],
    deps = [
        ":pjrt_client",
        ":semaphore",
        ":tracked_tfrt_cpu_device_buffer",
        ":utils",
        ":worker_thread",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:computation_placer_hdr",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_module_util",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/cpu:cpu_executable",
        "//tensorflow/compiler/xla/service/cpu:cpu_xfeed",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme",
        "//third_party/eigen3",  # TODO(zhangqiaorjc): Remove if use TFRT threadpool.
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "lru_cache",
    hdrs = ["lru_cache.h"],
    deps = [
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "lru_cache_test",
    srcs = ["lru_cache_test.cc"],
    deps = [
        ":lru_cache",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:logging",
    ],
)

cc_library(
    name = "transpose",
    srcs = [
        "transpose.cc",
        "transpose_kernels.h",
    ],
    hdrs = ["transpose.h"],
    deps = [
        ":lru_cache",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core/platform:logging",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "transpose_test",
    srcs = ["transpose_test.cc"],
    deps = [
        ":transpose",
        "//tensorflow/compiler/xla:array",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "//third_party/eigen3",
        "@com_google_absl//absl/numeric:int128",
    ],
)
