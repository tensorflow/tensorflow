package(
    default_visibility = [
        "//tensorflow/compiler/tf2xla:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

XLA_OPS_DEPS = [
    "@com_google_absl//absl/container:flat_hash_map",
    "@com_google_absl//absl/memory",
    "//tensorflow/compiler/jit:common",
    "//tensorflow/compiler/jit:compilation_passes",
    "//tensorflow/compiler/jit:flags",
    "//tensorflow/compiler/jit:xla_activity_listener",
    "//tensorflow/compiler/jit:xla_activity_proto_cc",
    "//tensorflow/compiler/jit:xla_compilation_cache",
    "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
    "//tensorflow/compiler/jit:xla_cluster_util",
    "//tensorflow/compiler/jit:xla_launch_util",
    "//tensorflow/compiler/tf2xla:common",
    "//tensorflow/compiler/tf2xla:tf2xla_util",
    "//tensorflow/compiler/tf2xla:xla_compiler",
    "//tensorflow/compiler/xla:executable_run_options",
    "//tensorflow/compiler/xla:status_macros",
    "//tensorflow/compiler/xla:statusor",
    "//tensorflow/compiler/xla/client:client_library",
    "//tensorflow/compiler/xla/client:local_client",
    "//tensorflow/compiler/xla/service:compiler",
    "//tensorflow/core:core_cpu_internal",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:state_ops_op_lib",
    "//tensorflow/core:stream_executor_no_cuda",
    "//tensorflow/core/profiler/lib:traceme",
    "//tensorflow/stream_executor:tf_allocator_adapter",
]

# Linked by tensorflow core, without registration of jit compilation passes.
cc_library(
    name = "xla_ops_no_jit_rewrite_registration",
    srcs = ["xla_ops.cc"],
    hdrs = ["xla_ops.h"],
    deps = XLA_OPS_DEPS,
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
