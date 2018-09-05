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
    default_visibility = [
        ":internal",
    ],
)

load("//tensorflow:tensorflow.bzl", "cc_header_only_library")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")

# Target that bundles up the XLA CPU and GPU JIT devices.
cc_library(
    name = "jit",
    visibility = [
        ":friends",
        "//learning/tfx:__subpackages__",
    ],
    deps = [
        ":xla_cpu_device",
        ":xla_cpu_jit",
        "//tensorflow/compiler/plugin",
    ] + if_cuda_is_configured([
        ":xla_gpu_device",
        ":xla_gpu_jit",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "xla_cpu_jit",
    visibility = ["//visibility:public"],
    deps = [
        ":jit_compilation_passes",
        "//tensorflow/compiler/jit/kernels:xla_launch_op",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla/service:cpu_plugin",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_gpu_jit",
    visibility = ["//visibility:public"],
    deps = if_cuda([
        ":jit_compilation_passes",
        "//tensorflow/compiler/jit/kernels:xla_launch_op",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla/service:gpu_plugin",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "xla_cpu_device",
    srcs = ["xla_cpu_device.cc"],
    visibility = [":friends"],
    deps = [
        ":jit_compilation_passes",
        ":xla_device",
        "//tensorflow/compiler/jit/kernels:xla_launch_op",
        "//tensorflow/compiler/jit/legacy_flags:xla_device_flags",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla/service:cpu_plugin",  # buildcleaner: keep
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
        "//tensorflow/compiler/jit/kernels:xla_launch_op",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla/service:gpu_plugin",  # buildcleaner: keep
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_interpreter_device",
    srcs = ["xla_interpreter_device.cc"],
    visibility = [":friends"],
    deps = [
        ":jit_compilation_passes",
        ":xla_device",
        "//tensorflow/compiler/jit/kernels:xla_launch_op",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla/service:interpreter_plugin",  # buildcleaner: keep
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_tensor",
    srcs = ["xla_tensor.cc"],
    hdrs = ["xla_tensor.h"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "xla_device",
    srcs = [
        "xla_compile_on_demand_op.cc",
        "xla_device.cc",
        "xla_device_context.cc",
        "xla_device_ops.cc",
    ],
    hdrs = [
        "xla_compile_on_demand_op.h",
        "xla_device.h",
        "xla_device_context.h",
        "xla_device_ops.h",
    ],
    # Public visibility is needed for external TF/XLA backends.
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":jit_compilation_passes",
        ":xla_launch_util",
        ":xla_tensor",
        "//tensorflow/compiler/jit/ops:xla_ops",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:dump_graph",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:stream_pool",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:fifo_queue",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:identity_n_op",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:no_op",
        "//tensorflow/core/kernels:queue_op",
        "//tensorflow/core/kernels:resource_variable_ops",
        "//tensorflow/core/kernels:sendrecv_ops",
        "//tensorflow/core/kernels:shape_ops",
        "//tensorflow/core/kernels:variable_ops",
        "//tensorflow/core/kernels/data:generator_dataset_op",
        "//tensorflow/core/kernels/data:iterator_ops",
        "//tensorflow/core/kernels/data:prefetch_dataset_op",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "shape_inference_helpers",
    srcs = ["shape_inference_helpers.cc"],
    hdrs = ["shape_inference_helpers.h"],
    deps = ["//tensorflow/core:graph"],
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
    name = "xla_launch_util",
    srcs = ["xla_launch_util.cc"],
    hdrs = ["xla_launch_util.h"],
    deps = [
        ":common",
        ":xla_compilation_cache",
        ":xla_tensor",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:device_memory_allocator",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:variable_ops",
        "@com_google_absl//absl/memory",
    ],
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
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:variable_ops",
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
    name = "create_xla_launch_op",
    srcs = [
        "create_xla_launch_op.cc",
        "create_xla_launch_op.h",
    ],
    deps = [
        ":common",
        ":compilation_passes",
        "//tensorflow/compiler/jit/kernels:xla_launch_op",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "create_xla_launch_op_test",
    srcs = [
        "create_xla_launch_op.h",
        "create_xla_launch_op_test.cc",
    ],
    deps = [
        ":create_xla_launch_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "resource_operation_safety_analysis",
    srcs = ["resource_operation_safety_analysis.cc"],
    hdrs = ["resource_operation_safety_analysis.h"],
    deps = [
        "//tensorflow/compiler/jit/graphcycles",
        "//tensorflow/compiler/tf2xla:resource_operation_table",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "resource_operation_safety_analysis_test",
    srcs = ["resource_operation_safety_analysis_test.cc"],
    deps = [
        ":common",
        ":resource_operation_safety_analysis",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/compiler/jit/kernels:xla_launch_op",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "compilation_passes",
    srcs = [
        "build_xla_launch_ops_pass.cc",
        "deadness_analysis.cc",
        "deadness_analysis_internal.h",
        "encapsulate_subgraphs_pass.cc",
        "mark_for_compilation_pass.cc",
        "mark_for_compilation_pass_test_helper.cc",
        "partially_decluster_pass.cc",
    ],
    hdrs = [
        "build_xla_launch_ops_pass.h",
        "deadness_analysis.h",
        "encapsulate_subgraphs_pass.h",
        "mark_for_compilation_pass.h",
        "mark_for_compilation_pass_test_helper.h",
        "partially_decluster_pass.h",
    ],
    deps = [
        ":common",
        ":shape_inference_helpers",
        ":union_find",
        ":xla_cluster_util",
        "//tensorflow/compiler/jit/graphcycles",
        "//tensorflow/compiler/jit/legacy_flags:mark_for_compilation_pass_flags",
        "//tensorflow/compiler/jit/ops:xla_ops",
        "//tensorflow/compiler/tf2xla:dump_graph",
        "//tensorflow/compiler/tf2xla:resource_operation_table",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:bounds_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "xla_cluster_util",
    srcs = ["xla_cluster_util.cc"],
    hdrs = ["xla_cluster_util.h"],
    deps = [
        ":resource_operation_safety_analysis",
        "//tensorflow/compiler/jit/graphcycles",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:bounds_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "union_find",
    hdrs = ["union_find.h"],
)

cc_library(
    name = "producer_consumer_queue",
    hdrs = ["producer_consumer_queue.h"],
    deps = ["//tensorflow/core:lib"],
)

tf_cc_test(
    name = "producer_consumer_queue_test",
    size = "small",
    srcs = ["producer_consumer_queue_test.cc"],
    deps = [
        ":producer_consumer_queue",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "deadness_analysis_test",
    size = "small",
    srcs = [
        "deadness_analysis_internal.h",
        "deadness_analysis_test.cc",
    ],
    deps = [
        ":common",
        ":compilation_passes",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/compiler/jit/kernels:xla_launch_op",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "compilation_passes_test",
    size = "small",
    srcs = [
        "encapsulate_subgraphs_pass_test.cc",
        "mark_for_compilation_pass_test.cc",
        "partially_decluster_pass_test.cc",
    ],
    deps = [
        ":common",
        ":compilation_passes",
        ":xla_cluster_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/compiler/jit/kernels:xla_launch_op",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "xla_cluster_util_test",
    size = "small",
    srcs = [
        "xla_cluster_util_test.cc",
    ],
    deps = [
        ":common",
        ":xla_cluster_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/compiler/jit/kernels:xla_launch_op",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "xla_launch_util_test",
    size = "small",
    srcs = ["xla_launch_util_test.cc"],
    deps = [
        ":common",
        ":xla_compilation_cache",
        ":xla_launch_util",
        ":xla_tensor",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/kernels:variable_ops",
    ],
)

cc_library(
    name = "xla_fusion_optimizer",
    srcs = ["xla_fusion_optimizer.cc"],
    hdrs = ["xla_fusion_optimizer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":compilation_passes",
        ":union_find",
        ":xla_cluster_util",
        "//tensorflow/compiler/jit/graphcycles",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "xla_fusion_optimizer_test",
    srcs = ["xla_fusion_optimizer_test.cc"],
    deps = [
        ":common",
        ":xla_cluster_util",
        ":xla_fusion_optimizer",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/core:graph",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler/utils:grappler_test",
    ],
)

# This target can be used by XLA device plugins to prevent circular dependencies, and provides access to all of the required headers for building a device library.
cc_header_only_library(
    name = "xla_jit_headers_lib",
    visibility = ["//visibility:public"],
    deps = [
        ":xla_cpu_device",
        ":xla_cpu_jit",
        ":xla_gpu_device",
        ":xla_gpu_jit",
    ],
)
