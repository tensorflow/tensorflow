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

# NB! Removing the cc_header_only_library import breaks the OSS build since
# copybara injects some build rules that use it.
load("//tensorflow:tensorflow.bzl", "cc_header_only_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

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
    ] + if_cuda([
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
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_cpu_only_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_dummy_ops",
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
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_dummy_ops",
        "//tensorflow/compiler/xla/service:gpu_plugin",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "xla_cpu_device",
    srcs = ["xla_cpu_device.cc"],
    visibility = [":friends"],
    deps = [
        ":flags",
        ":jit_compilation_passes",
        ":xla_device",
        ":xla_kernel_creator",  # buildcleaner: keep
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla/service:cpu_plugin",  # buildcleaner: keep
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
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
        ":xla_kernel_creator",  # buildcleaner: keep
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla/service:gpu_plugin",  # buildcleaner: keep
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
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
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla/service:interpreter_plugin",  # buildcleaner: keep
        "@com_google_absl//absl/memory",
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
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:stream_pool",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:control_flow_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:no_op_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:resource_variable_ops_op_lib",
        "//tensorflow/core:sendrecv_ops_op_lib",
        "//tensorflow/core:state_ops_op_lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:fifo_queue",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:host_constant_op",
        "//tensorflow/core/kernels:identity_n_op",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:no_op",
        "//tensorflow/core/kernels:queue_op",
        "//tensorflow/core/kernels:resource_variable_ops",
        "//tensorflow/core/kernels:sendrecv_ops",
        "//tensorflow/core/kernels:shape_ops",
        "//tensorflow/core/kernels:stack",
        "//tensorflow/core/kernels:variable_ops",
        "//tensorflow/core/kernels/data:generator_dataset_op",
        "//tensorflow/core/kernels/data:iterator_ops",
        "//tensorflow/core/kernels/data:optional_ops",
        "//tensorflow/core/kernels/data:prefetch_dataset_op",
        "//tensorflow/stream_executor/platform",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
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
    name = "flags",
    srcs = ["flags.cc"],
    hdrs = ["flags.h"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/compiler/xla:parse_flags_from_env",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

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
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "xla_compilation_cache",
    srcs = ["xla_compilation_cache.cc"],
    hdrs = ["xla_compilation_cache.h"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "xla_compilation_cache_test",
    srcs = [
        "xla_compilation_cache_test.cc",
    ],
    deps = [
        ":xla_compilation_cache",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "jit_compilation_passes",
    srcs = ["jit_compilation_pass_registration.cc"],
    deps = [
        ":compilation_passes",
        "//tensorflow/compiler/tf2xla:functionalize_control_flow_pass_registration",
        "//tensorflow/core:core_cpu_internal",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_kernel_creator",
    srcs = [
        "xla_kernel_creator.cc",
        "xla_kernel_creator.h",
    ],
    deps = [
        ":common",
        ":compilation_passes",
        "//tensorflow/compiler/jit/kernels:xla_ops",
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
    name = "xla_kernel_creator_test",
    srcs = [
        "xla_kernel_creator.h",
        "xla_kernel_creator_test.cc",
    ],
    deps = [
        ":xla_kernel_creator",
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
        "@com_google_absl//absl/container:flat_hash_set",
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
        "//tensorflow/compiler/jit/kernels:xla_ops",
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
    name = "shape_inference",
    srcs = ["shape_inference.cc"],
    hdrs = ["shape_inference.h"],
    deps = [
        ":shape_inference_helpers",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        ":shape_inference",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "shape_inference_test",
    srcs = ["shape_inference_test.cc"],
    deps = [
        ":shape_inference",
        ":test_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/kernels:constant_op",
    ],
)

cc_library(
    name = "encapsulate_util",
    srcs = ["encapsulate_util.cc"],
    hdrs = ["encapsulate_util.h"],
    deps = [
        ":shape_inference",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "encapsulate_util_test",
    srcs = ["encapsulate_util_test.cc"],
    deps = [
        ":encapsulate_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "compilation_passes",
    srcs = [
        "build_xla_ops_pass.cc",
        "clone_constants_for_better_clustering.cc",
        "deadness_analysis.cc",
        "deadness_analysis_internal.h",
        "encapsulate_subgraphs_pass.cc",
        "encapsulate_xla_computations_pass.cc",
        "extract_outside_compilation_pass.cc",
        "increase_dynamism_for_auto_jit_pass.cc",
        "introduce_floating_point_jitter_pass.cc",
        "mark_for_compilation_pass.cc",
        "mark_for_compilation_pass_test_helper.cc",
        "partially_decluster_pass.cc",
    ],
    hdrs = [
        "build_xla_ops_pass.h",
        "clone_constants_for_better_clustering.h",
        "deadness_analysis.h",
        "encapsulate_subgraphs_pass.h",
        "encapsulate_xla_computations_pass.h",
        "extract_outside_compilation_pass.h",
        "increase_dynamism_for_auto_jit_pass.h",
        "introduce_floating_point_jitter_pass.h",
        "mark_for_compilation_pass.h",
        "mark_for_compilation_pass_test_helper.h",
        "partially_decluster_pass.h",
    ],
    deps = [
        ":common",
        ":device_info_cache",
        ":encapsulate_util",
        ":flags",
        ":shape_inference_helpers",
        ":union_find",
        ":xla_cluster_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:scope_internal",
        "//tensorflow/compiler/jit/graphcycles",
        "//tensorflow/compiler/jit/ops:xla_ops",
        "//tensorflow/compiler/tf2xla:resource_operation_table",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/cc:xla_jit_ops",
        "//tensorflow/compiler/tf2xla/cc:xla_ops",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_bounds_check",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "xla_cluster_util",
    srcs = ["xla_cluster_util.cc"],
    hdrs = ["xla_cluster_util.h"],
    deps = [
        ":flags",
        ":resource_operation_safety_analysis",
        "//tensorflow/compiler/jit/graphcycles",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_bounds_check",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "device_info_cache",
    srcs = ["device_info_cache.cc"],
    hdrs = ["device_info_cache.h"],
    deps = [
        ":xla_cluster_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "union_find",
    hdrs = ["union_find.h"],
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
        "//tensorflow/compiler/jit/kernels:xla_ops",
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
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "compilation_passes_test_main",
    testonly = True,
    srcs = ["compilation_passes_test_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":flags",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "compilation_passes_test",
    size = "small",
    srcs = [
        "build_xla_ops_pass_test.cc",
        "clone_constants_for_better_clustering_test.cc",
        "encapsulate_subgraphs_pass_test.cc",
        "encapsulate_xla_computations_pass_test.cc",
        "extract_outside_compilation_pass_test.cc",
        "increase_dynamism_for_auto_jit_pass_test.cc",
        "introduce_floating_point_jitter_pass_internal.h",
        "introduce_floating_point_jitter_pass_test.cc",
        "mark_for_compilation_pass_test.cc",
        "partially_decluster_pass_test.cc",
    ],
    deps = [
        ":common",
        ":compilation_passes",
        ":compilation_passes_test_main",
        ":encapsulate_util",
        ":flags",
        ":node_matchers",
        ":xla_cluster_util",
        ":xla_cpu_device",
        ":xla_gpu_device",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/compiler/tf2xla:test_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/cc:xla_jit_ops",
        "//tensorflow/compiler/tf2xla/cc:xla_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_dummy_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
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
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "node_matchers",
    testonly = True,
    srcs = ["node_matchers.cc"],
    hdrs = ["node_matchers.h"],
    deps = [
        "//tensorflow/cc:ops",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "node_matchers_test",
    srcs = ["node_matchers_test.cc"],
    deps = [
        ":node_matchers",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:ops",
        "//tensorflow/core:ops",
        "//tensorflow/core:test_main",
    ],
)

tf_custom_op_py_library(
    name = "xla_ops_py",
    kernels = ["//tensorflow/compiler/jit/ops:xla_ops"],
    visibility = [
        ":friends",
    ],
    deps = [
        "//tensorflow/compiler/jit/ops:xla_ops_grad",
        "//tensorflow/compiler/jit/ops:xla_ops_wrapper_py",
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
