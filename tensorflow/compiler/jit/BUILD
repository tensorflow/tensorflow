licenses(["notice"])  # Apache 2.0

package_group(
    name = "internal",
    includes = [
        "//tensorflow/compiler/tf2xla:internal",
    ],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/tf2xla:friends",
    ],
)

package(
    default_visibility = [":internal"],
)

load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

# Target that bundles up the XLA CPU and GPU JIT devices.
cc_library(
    name = "jit",
    visibility = [":friends"],
    deps = [
        ":xla_cpu_device",
        ":xla_cpu_jit",
        ":xla_gpu_device",
        ":xla_gpu_jit",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_cpu_jit",
    visibility = [":friends"],
    deps = [
        ":jit_compilation_passes",
        ":xla_local_launch_op",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla/service:cpu_plugin",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_gpu_jit",
    visibility = [":friends"],
    deps = [
        ":jit_compilation_passes",
        ":xla_local_launch_op",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla/service:gpu_plugin",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_cpu_device",
    srcs = ["xla_cpu_device.cc"],
    visibility = [":friends"],
    deps = [
        ":jit_compilation_passes",
        ":xla_device",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_gpu_device",
    srcs = ["xla_gpu_device.cc"],
    visibility = [":friends"],
    deps = [
        ":jit_compilation_passes",
        ":xla_device",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

# Internal targets below this point.

cc_library(
    name = "common",
    srcs = [
        "defs.cc",
    ],
    hdrs = [
        "defs.h",
    ],
    visibility = [":friends"],
)

cc_library(
    name = "xla_device",
    srcs = [
        "xla_device.cc",
        "xla_device_context.cc",
        "xla_device_launch_op.cc",
        "xla_device_ops.cc",
    ],
    hdrs = [
        "xla_device.h",
        "xla_device_context.h",
        "xla_device_launch_op.h",
        "xla_device_ops.h",
    ],
    deps = [
        ":common",
        ":jit_compilation_passes",
        ":xla_compilation_cache",
        ":xla_local_launch_op",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:dump_graph",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core:tensorflow_opensource",
        "//tensorflow/core/kernels:assign_op",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:no_op",
        "//tensorflow/core/kernels:sendrecv_ops",
        "//tensorflow/core/kernels:variable_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_compilation_cache",
    srcs = ["xla_compilation_cache.cc"],
    hdrs = ["xla_compilation_cache.h"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:dump_graph",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "jit_compilation_passes",
    srcs = ["jit_compilation_pass_registration.cc"],
    deps = [
        ":compilation_passes",
        "//tensorflow/core:core_cpu_internal",
    ],
    alwayslink = 1,
)

cc_library(
    name = "compilation_passes",
    srcs = [
        "build_xla_launch_ops_pass.cc",
        "encapsulate_subgraphs_pass.cc",
        "graph_to_functiondef.cc",
        "mark_for_compilation_pass.cc",
    ],
    hdrs = [
        "build_xla_launch_ops_pass.h",
        "encapsulate_subgraphs_pass.h",
        "graph_to_functiondef.h",
        "mark_for_compilation_pass.h",
    ],
    deps = [
        ":common",
        ":parallel_check_op",
        ":xla_local_launch_op",
        "//tensorflow/compiler/jit/graphcycles",
        "//tensorflow/compiler/jit/legacy_flags:encapsulate_subgraphs_pass_flags",
        "//tensorflow/compiler/jit/legacy_flags:mark_for_compilation_pass_flags",
        "//tensorflow/compiler/tf2xla:const_analysis",
        "//tensorflow/compiler/tf2xla:dump_graph",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "compilation_passes_test",
    size = "small",
    srcs = [
        "encapsulate_subgraphs_pass_test.cc",
        "graph_to_functiondef_test.cc",
        "mark_for_compilation_pass_test.cc",
    ],
    deps = [
        ":compilation_passes",
        ":xla_local_launch_op",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "xla_local_launch_op",
    srcs = ["xla_local_launch_op.cc"],
    hdrs = ["xla_local_launch_op.h"],
    deps = [
        ":common",
        ":xla_compilation_cache",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_local_runtime_context",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core:tensorflow_opensource",
    ],
    alwayslink = 1,
)

tf_kernel_library(
    name = "parallel_check_op",
    srcs = ["parallel_check_op.cc"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/compiler/jit/legacy_flags:parallel_check_op_flags",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

# -----------------------------------------------------------------------------

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
