licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

# Platform specific build config
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "if_static",
)

cc_library(
    name = "static_schedule",
    srcs = ["static_schedule.cc"],
    hdrs = [
        "static_schedule.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/costs:cost_estimator",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/costs:op_level_cost_estimator",
        "//tensorflow/core/grappler/costs:virtual_placer",
    ],
)

tf_cuda_cc_test(
    name = "static_schedule_test",
    srcs = ["static_schedule_test.cc"],
    deps = [
        ":static_schedule",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "auto_parallel",
    srcs = ["auto_parallel.cc"],
    hdrs = [
        "auto_parallel.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:devices",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
    ],
)

tf_cuda_cc_test(
    name = "auto_parallel_test",
    srcs = ["auto_parallel_test.cc"],
    deps = [
        ":auto_parallel",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
    ],
)

cc_library(
    name = "constant_folding",
    srcs = ["constant_folding.cc"],
    hdrs = [
        "constant_folding.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":evaluation_utils",
        ":graph_optimizer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/utils:symbolic_shapes",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "constant_folding_test",
    srcs = ["constant_folding_test.cc"],
    shard_count = 5,
    deps = [
        ":constant_folding",
        ":dependency_optimizer",
        "//tensorflow/cc:array_ops_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "function_optimizer",
    srcs = ["function_optimizer.cc"],
    hdrs = [
        "function_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:graph_topology_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/utils:functions",
        "//tensorflow/core/grappler/utils:traversal",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "function_optimizer_test",
    srcs = ["function_optimizer_test.cc"],
    deps = [
        ":function_optimizer",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/utils:grappler_test",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "graph_optimizer",
    hdrs = [
        "graph_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "graph_optimizer_stage",
    srcs = ["graph_optimizer_stage.cc"],
    hdrs = ["graph_optimizer_stage.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/costs:graph_properties",
    ],
)

tf_cuda_cc_test(
    name = "graph_optimizer_stage_test",
    size = "small",
    srcs = ["graph_optimizer_stage_test.cc"],
    deps = [
        ":graph_optimizer_stage",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "custom_graph_optimizer",
    hdrs = [
        "custom_graph_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "arithmetic_optimizer",
    srcs = ["arithmetic_optimizer.cc"],
    hdrs = [
        "arithmetic_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":constant_folding",
        ":graph_optimizer",
        ":graph_optimizer_stage",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:graph_topology_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/utils:symbolic_shapes",
        "//tensorflow/core/grappler/utils:topological_sort",
        "//tensorflow/core/grappler/utils:traversal",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cuda_cc_test(
    name = "arithmetic_optimizer_test",
    size = "small",
    srcs = ["arithmetic_optimizer_test.cc"],
    deps = [
        ":arithmetic_optimizer",
        ":constant_folding",
        ":model_pruner",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//tensorflow/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "dependency_optimizer",
    srcs = ["dependency_optimizer.cc"],
    hdrs = [
        "dependency_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":constant_folding",
        ":graph_optimizer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/utils:topological_sort",
    ],
)

tf_cuda_cc_test(
    name = "dependency_optimizer_test",
    size = "small",
    srcs = ["dependency_optimizer_test.cc"],
    deps = [
        ":constant_folding",
        ":dependency_optimizer",
        ":model_pruner",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//tensorflow/core/grappler/utils:grappler_test",
        "//tensorflow/core/grappler/utils:topological_sort",
    ],
)

cc_library(
    name = "model_pruner",
    srcs = ["model_pruner.cc"],
    hdrs = [
        "model_pruner.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cuda_cc_test(
    name = "model_pruner_test",
    srcs = ["model_pruner_test.cc"],
    deps = [
        ":model_pruner",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:devices",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//tensorflow/core/grappler/utils:grappler_test",
    ],
)

tf_kernel_library(
    name = "gpu_swapping_kernels",
    srcs = [
        "gpu_swapping_kernels.cc",
    ],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "gpu_swapping_ops",
    srcs = [
        "gpu_swapping_ops.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "memory_optimizer",
    srcs = [
        "memory_optimizer.cc",
    ],
    hdrs = [
        "memory_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        ":static_schedule",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:graph_topology_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/costs:graph_memory",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/costs:utils",
        "//tensorflow/core/grappler/utils:topological_sort",
        "//tensorflow/core/grappler/utils:traversal",
    ],
)

tf_cuda_cc_test(
    name = "memory_optimizer_test",
    srcs = ["memory_optimizer_test.cc"],
    tags = ["no_cuda_on_cpu_tap"],  # Do not re-enable again without actually testing.
    deps = [
        ":gpu_swapping_kernels",
        ":gpu_swapping_ops",
        ":memory_optimizer",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "layout_optimizer",
    srcs = ["layout_optimizer.cc"],
    hdrs = [
        "layout_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:devices",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/costs:virtual_placer",
        "//tensorflow/core/grappler/utils:frame",
    ],
)

tf_cuda_cc_test(
    name = "layout_optimizer_test",
    srcs = ["layout_optimizer_test.cc"],
    deps = [
        ":layout_optimizer",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:devices",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:single_machine",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/costs:virtual_placer",
        "//tensorflow/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "meta_optimizer",
    srcs = ["meta_optimizer.cc"],
    hdrs = [
        "meta_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":arithmetic_optimizer",
        ":auto_parallel",
        ":constant_folding",
        ":custom_graph_optimizer_registry",
        ":debug_stripper",
        ":dependency_optimizer",
        ":experimental_implementation_selector",
        ":function_optimizer",
        ":graph_optimizer",
        ":layout_optimizer",
        ":loop_optimizer",
        ":memory_optimizer",
        ":model_pruner",
        ":pin_to_host_optimizer",
        ":remapper",
        ":scoped_allocator_optimizer",
        ":shape_optimizer",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/utils:colocation",
        "//tensorflow/core/grappler/utils:functions",
        "//tensorflow/core/grappler/utils:topological_sort",
        "//tensorflow/core/grappler/verifiers:graph_verifier",
        "//tensorflow/core/grappler/verifiers:structure_verifier",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "meta_optimizer_test",
    srcs = ["meta_optimizer_test.cc"],
    deps = [
        ":custom_graph_optimizer",
        ":custom_graph_optimizer_registry",
        ":meta_optimizer",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//tensorflow/core/grappler/utils:grappler_test",
        "@com_google_absl//absl/strings",
    ],
)

# This rule is header-only unless the build is static (--config=monolithic). Its
# implementation is included directly in the framework shared object.
cc_library(
    name = "custom_graph_optimizer_registry",
    hdrs = ["custom_graph_optimizer_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":custom_graph_optimizer",
        "//tensorflow/core:lib",
    ] + if_static(
        [":custom_graph_optimizer_registry_impl"],
    ),
)

# This rule contains static variables for the optimizer registry. Do not depend
# on it directly; use :custom_graph_optimizer_registry, and link against
# libtensorflow_framework.so for the registry symbols.
cc_library(
    name = "custom_graph_optimizer_registry_impl",
    srcs = ["custom_graph_optimizer_registry.cc"],
    hdrs = ["custom_graph_optimizer_registry.h"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":custom_graph_optimizer",
        "//tensorflow/core:lib",
    ],
)

tf_cuda_cc_test(
    name = "custom_graph_optimizer_registry_test",
    size = "small",
    srcs = ["custom_graph_optimizer_registry_test.cc"],
    deps = [
        ":custom_graph_optimizer",
        ":custom_graph_optimizer_registry",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "loop_optimizer",
    srcs = ["loop_optimizer.cc"],
    hdrs = [
        "loop_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":constant_folding",
        ":evaluation_utils",
        ":graph_optimizer",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:graph_topology_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/utils:frame",
        "//tensorflow/core/grappler/utils:traversal",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cuda_cc_test(
    name = "loop_optimizer_test",
    srcs = ["loop_optimizer_test.cc"],
    deps = [
        ":loop_optimizer",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//tensorflow/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "shape_optimizer",
    srcs = ["shape_optimizer.cc"],
    hdrs = [
        "shape_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/utils:frame",
        "//tensorflow/core/grappler/utils:symbolic_shapes",
    ],
)

tf_cc_test(
    name = "shape_optimizer_test",
    srcs = ["shape_optimizer_test.cc"],
    deps = [
        ":shape_optimizer",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/utils:grappler_test",
    ],
)

tf_kernel_library(
    name = "remapper",
    srcs = ["remapper.cc"],
    hdrs = [
        "remapper.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":constant_folding",
        ":graph_optimizer",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/utils:symbolic_shapes",
        "//tensorflow/core/grappler/utils:topological_sort",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cuda_cc_test(
    name = "remapper_test",
    srcs = ["remapper_test.cc"],
    deps = [
        ":remapper",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:devices",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//tensorflow/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "debug_stripper",
    srcs = ["debug_stripper.cc"],
    hdrs = [
        "debug_stripper.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:graph_optimizer",
    ],
)

tf_cuda_cc_test(
    name = "debug_stripper_test",
    size = "small",
    srcs = ["debug_stripper_test.cc"],
    deps = [
        ":debug_stripper",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "scoped_allocator_optimizer",
    srcs = ["scoped_allocator_optimizer.cc"],
    hdrs = [
        "scoped_allocator_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/utils:frame",
    ],
)

tf_cc_test(
    name = "scoped_allocator_optimizer_test",
    size = "small",
    srcs = ["scoped_allocator_optimizer_test.cc"],
    deps = [
        ":scoped_allocator_optimizer",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "evaluation_utils",
    srcs = ["evaluation_utils.cc"],
    hdrs = [
        "evaluation_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "evaluation_utils_test",
    srcs = ["evaluation_utils_test.cc"],
    deps = [
        ":evaluation_utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "function_api_info",
    srcs = ["function_api_info.cc"],
    hdrs = ["function_api_info.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "function_api_info_test",
    size = "small",
    srcs = ["function_api_info_test.cc"],
    deps = [
        ":function_api_info",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "experimental_implementation_selector",
    srcs = ["experimental_implementation_selector.cc"],
    hdrs = ["experimental_implementation_selector.h"],
    deps = [
        ":custom_graph_optimizer",
        ":custom_graph_optimizer_registry",
        ":function_api_info",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler/costs:graph_properties",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "experimental_implementation_selector_test",
    size = "small",
    srcs = ["experimental_implementation_selector_test.cc"],
    deps = [
        ":custom_graph_optimizer",
        ":custom_graph_optimizer_registry",
        ":experimental_implementation_selector",
        ":function_api_info",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//tensorflow/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "pin_to_host_optimizer",
    srcs = ["pin_to_host_optimizer.cc"],
    hdrs = [
        "pin_to_host_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/utils:frame",
        "//tensorflow/core/grappler/utils:symbolic_shapes",
        "//tensorflow/core/grappler/utils:topological_sort",
    ],
)

tf_cuda_cc_test(
    name = "pin_to_host_optimizer_test",
    srcs = ["pin_to_host_optimizer_test.cc"],
    deps = [
        ":pin_to_host_optimizer",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/utils:grappler_test",
    ],
)
