load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/tf2xla:internal",
        "//tensorflow/core/tpu:__subpackages__",
    ],
    licenses = ["notice"],
)

XLA_OPS_DEPS = [
    "@com_google_absl//absl/container:flat_hash_map",
    "@com_google_absl//absl/memory",
    "@com_google_absl//absl/synchronization",
    "//tensorflow/compiler/jit:common",
    "//tensorflow/compiler/jit:compilation_passes",
    "//tensorflow/compiler/jit:flags",
    "//tensorflow/compiler/jit:xla_activity_listener",
    "//tensorflow/compiler/jit:xla_activity_proto_cc",
    "//tensorflow/compiler/jit:device_compiler",
    "//tensorflow/compiler/jit:variable_info",
    "//tensorflow/compiler/jit:variable_info_util",
    "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
    "//tensorflow/compiler/jit:xla_cluster_util",
    "//tensorflow/compiler/jit:xla_host_recv_device_context",
    "//tensorflow/compiler/jit:xla_launch_util",
    "//tensorflow/compiler/tf2xla:common",
    "//tensorflow/compiler/tf2xla:tf2xla_util",
    "//tensorflow/compiler/tf2xla:xla_compiler",
    "//tensorflow/compiler/tf2xla:xla_helpers",
    "//tensorflow/compiler/tf2xla:xla_op_registry",
    "//tensorflow/compiler/xla:executable_run_options",
    "//tensorflow/compiler/xla:status_macros",
    "//tensorflow/compiler/xla:statusor",
    "//tensorflow/compiler/xla/client:client_library",
    "//tensorflow/compiler/xla/client:local_client",
    "//tensorflow/compiler/xla/service:compiler",
    "//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
    "//tensorflow/core:core_cpu_internal",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:state_ops_op_lib",
    "//tensorflow/core/platform:stream_executor_no_cuda",
    "//tensorflow/core/profiler/lib:traceme",
    "//tensorflow/compiler/xla/stream_executor:tf_allocator_adapter",
    "@com_google_absl//absl/types:optional",
]

# Linked by tensorflow core, without registration of jit compilation passes.
cc_library(
    name = "xla_ops_no_jit_rewrite_registration",
    srcs = ["xla_ops.cc"],
    hdrs = ["xla_ops.h"],
    deps = XLA_OPS_DEPS + [
        "//tensorflow/compiler/jit:device_compilation_cache",
        "//tensorflow/compiler/jit:device_compilation_profiler",
        "//tensorflow/compiler/jit:tf_graph_to_hlo_compiler",
        "//tensorflow/compiler/jit:tf_to_hlo_compiler",
        "//tensorflow/compiler/jit:xla_compile_util",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/core/platform:refcount",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_ops",
    hdrs = ["xla_ops.h"],
    deps = XLA_OPS_DEPS + [
        ":xla_ops_no_jit_rewrite_registration",
        "//tensorflow/compiler/jit:jit_compilation_passes",
    ],
    alwayslink = 1,
)
