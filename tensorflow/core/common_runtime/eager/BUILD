load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cc_test_mkl",
    "tf_copts",
    "tf_cuda_library",
    "tf_mkl_kernel_library",
)
load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl",
)

package(
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    licenses = ["notice"],
)

# TODO(b/152902651): Remove this file once all circular dependencies are resolved.
tf_cuda_library(
    name = "core",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":core_no_xla",
        "//tensorflow/compiler/jit:xla_kernel_creator",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "core_no_xla",
    srcs = [
        "core.cc",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":eager_operation",
        ":execute",
        ":placement_utils",
        ":tensor_handle",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/c/eager:abstract_function",
        "//tensorflow/core/platform:errors",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "eager_executor",
    srcs = [
        "eager_executor.cc",
    ],
    hdrs = [
        "eager_executor.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:core_cpu_lib",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
        ],
    }),
)

tf_cuda_library(
    name = "context",
    srcs = [
        "context.cc",
    ],
    hdrs = [
        "context.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":eager_executor",
        ":kernel_and_device",
        ":custom_device",
        "@com_google_absl//absl/container:flat_hash_map",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_distributed_manager",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/core/distributed_runtime:rendezvous_mgr_interface",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/nccl:collective_communicator",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "@com_google_absl//absl/types:optional",
            "//tensorflow/core:core_cpu_lib",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:session_options",
            "//tensorflow/core/distributed_runtime:collective_param_resolver_distributed",
            "//tensorflow/core/distributed_runtime:session_mgr",
            "//tensorflow/core/distributed_runtime:device_resolver_distributed",
            "//tensorflow/core/distributed_runtime:rpc_collective_executor_mgr",
            "//tensorflow/core/distributed_runtime:worker_cache",
            "//tensorflow/core/distributed_runtime:server_lib",
            "//tensorflow/core/distributed_runtime:worker_session",
            "//tensorflow/core/distributed_runtime/eager:eager_client",
        ],
    }),
)

tf_cuda_library(
    name = "custom_device",
    srcs = [
        "custom_device.cc",
        "custom_device_op_handler.cc",
    ],
    hdrs = [
        "custom_device.h",
        "custom_device_op_handler.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/c/eager:immediate_execution_context",
            "//tensorflow/c/eager:immediate_execution_tensor_handle",
            "//tensorflow/c/eager:immediate_execution_operation",
            "//tensorflow/core/lib/core:status",
        ],
    }),
)

tf_cc_test(
    name = "custom_device_test",
    srcs = ["custom_device_test.cc"],
    deps = [
        ":context",
        ":core",
        ":custom_device",
        ":eager_operation",
        ":placement_utils",
        ":tensor_handle",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/kernels:resource_variable_ops",
        "//tensorflow/core/ops:resource_variable_ops_op_lib",
    ],
)

tf_cuda_library(
    name = "context_distributed_manager",
    srcs = [
        "context_distributed_manager.cc",
    ],
    hdrs = [
        "context_distributed_manager.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":eager_executor",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_distributed_manager",
        "//tensorflow/core/distributed_runtime:rendezvous_mgr_interface",
        "//tensorflow/core/distributed_runtime:master_env",
        "//tensorflow/core/distributed_runtime:worker_env",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core/distributed_runtime/coordination:coordination_service",
            "//tensorflow/core/distributed_runtime/coordination:coordination_service_agent",
            "//tensorflow/core/distributed_runtime/coordination:coordination_service_impl",
            "//tensorflow/core/distributed_runtime/preemption:preemption_notifier",
            "//tensorflow/core:core_cpu_lib",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:session_options",
            "//tensorflow/core/distributed_runtime:worker_cache",
            "//tensorflow/core/distributed_runtime:worker_interface",
            "//tensorflow/core/distributed_runtime:remote_device",
            "//tensorflow/core/distributed_runtime:server_lib",
            "//tensorflow/core/distributed_runtime:session_mgr",
            "//tensorflow/core/distributed_runtime:worker_session",
            "//tensorflow/core/distributed_runtime/eager:eager_client",
            "//tensorflow/core/distributed_runtime/eager:cluster_function_library_runtime",
            "//tensorflow/core/distributed_runtime/eager:remote_mgr",
        ],
    }),
)

tf_cc_test(
    name = "context_test",
    srcs = ["context_test.cc"],
    deps = [
        ":context",
        ":core",
        ":eager_operation",
        ":execute",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime/eager:context_distributed_manager",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/kernels:cast_op",
        "//tensorflow/core/kernels:logging_ops",
        "//tensorflow/core/kernels:math",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "placement_test",
    srcs = ["placement_test.cc"],
    deps = [
        ":context",
        ":core",
        ":eager_operation",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_library(
    name = "eager_operation",
    srcs = [
        "eager_operation.cc",
    ],
    hdrs = [
        "eager_operation.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":attr_builder",
        ":custom_device",
        ":context",
        ":eager_executor",
        ":kernel_and_device",
        ":tensor_handle",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/util:managed_stack_trace",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:core_cpu_lib",
        ],
    }),
)

tf_cc_test(
    name = "eager_operation_test",
    srcs = ["eager_operation_test.cc"],
    deps = [
        ":core",
        ":eager_operation",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cuda_library(
    name = "tensor_handle_data",
    srcs = [
        "tensor_handle_data.cc",
    ],
    hdrs = [
        "tensor_handle_data.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":eager_executor",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "@com_google_absl//absl/types:variant",
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core/profiler/lib:traceme",
        ],
    }),
)

tf_cuda_library(
    name = "tensor_handle",
    srcs = [
        "tensor_handle.cc",
    ],
    hdrs = [
        "tensor_handle.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":eager_executor",
        ":kernel_and_device",
        ":tensor_handle_data",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/types:variant",
            "//tensorflow/c:tf_tensor_internal",
            "//tensorflow/c/eager:immediate_execution_tensor_handle",
            "//tensorflow/core:core_cpu_lib",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:session_options",
            "//tensorflow/core/distributed_runtime/eager:remote_tensor_handle_data",
            "//tensorflow/core/profiler/lib:traceme",
        ],
    }),
)

tf_cc_test(
    name = "tensor_handle_test",
    srcs = ["tensor_handle_test.cc"],
    deps = [
        ":core",
        ":eager_operation",
        ":execute",
        ":tensor_handle",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

tf_cuda_library(
    name = "copy_to_device_node",
    hdrs = [
        "copy_to_device_node.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":eager_executor",
        ":tensor_handle",
    ] + select({
        "//tensorflow:android": [
        ],
        "//conditions:default": [
            "//tensorflow/core:core_cpu_lib",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:session_options",
            "//tensorflow/core/profiler/lib:scoped_memory_debug_annotation",
        ],
    }),
)

cc_library(
    name = "shape_inference",
    srcs = ["shape_inference.cc"],
    hdrs = ["shape_inference.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":tensor_handle",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

KERNEL_AND_DEVICE_DEPS = [
    "//tensorflow/core:core_cpu_lib",
    "//tensorflow/core:framework",
    "//tensorflow/core:framework_internal",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:protos_all_cc",
    "//tensorflow/core/profiler/lib:annotated_traceme",
    "//tensorflow/core/profiler/lib:traceme",
    "//tensorflow/core/grappler:grappler_item",
    "//tensorflow/core/grappler/optimizers:meta_optimizer",
]

tf_cuda_library(
    name = "kernel_and_device",
    srcs = [
        "kernel_and_device.cc",
    ],
    hdrs = [
        "kernel_and_device.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":attr_builder",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
        "@farmhash_archive//:farmhash",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": KERNEL_AND_DEVICE_DEPS,
    }),
)

tf_cc_test(
    name = "kernel_and_device_test",
    srcs = ["kernel_and_device_test.cc"],
    deps = [
        ":attr_builder",
        ":kernel_and_device",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:no_op_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:sendrecv_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:matmul_op",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "execute",
    srcs = [
        "execute.cc",
        "execute_node.cc",
    ],
    hdrs = [
        "execute.h",
        "execute_node.h",
    ] + if_mkl(["//tensorflow/core/graph:mkl_graph_util_header"]),
    copts = if_mkl(["-DINTEL_MKL"]),
    deps = [
        ":context",
        ":copy_to_device_node",
        ":eager_executor",
        ":eager_op_rewrite_registry",
        ":eager_operation",
        ":kernel_and_device",
        ":tensor_handle",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/memory",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/compiler/jit:common",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:scoped_memory_debug_annotation",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core/distributed_runtime/eager:remote_mgr",
            "//tensorflow/core:core_cpu_lib",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/distributed_runtime/eager:eager_client",
            "//tensorflow/core/distributed_runtime/eager:remote_execute_node",
            "//tensorflow/core/distributed_runtime/eager:remote_copy_node",
        ],
    }) + if_mkl([":mkl_eager_op_rewrite"]),
)

tf_cc_test(
    name = "execute_node_test",
    srcs = ["execute_node_test.cc"],
    deps = [
        ":context",
        ":core",
        ":execute",
        ":kernel_and_device",
        ":tensor_handle",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

tf_mkl_kernel_library(
    name = "mkl_eager_op_rewrite",
    srcs = ["mkl_eager_op_rewrite.cc"],
    deps = [
        ":eager_op_rewrite_registry",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core/graph:mkl_graph_util",
    ],
    alwayslink = 1,
)

tf_cc_test_mkl(
    name = "mkl_eager_op_rewrite_test",
    srcs = ["mkl_eager_op_rewrite_test.cc"],
    tags = [
        "nomac",
    ],
    deps = [
        ":core",
        ":eager_op_rewrite_registry",
        ":mkl_eager_op_rewrite",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:device_mgr",
    ],
)

cc_library(
    name = "eager_op_rewrite_registry",
    srcs = ["eager_op_rewrite_registry.cc"],
    hdrs = ["eager_op_rewrite_registry.h"],
    deps = [":eager_operation"],
)

tf_cc_test(
    name = "eager_op_rewrite_registry_test",
    srcs = ["eager_op_rewrite_registry_test.cc"],
    deps = [
        ":eager_op_rewrite_registry",
        ":mkl_core",
        "//tensorflow/core:lib",
        "//tensorflow/core:no_op_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# Temporary rule until the circular dependencies issue is resolved.
# TODO(mabuzain): remove this once original "core" package is fixed.
cc_library(
    name = "mkl_core",
    srcs = [
        "core.cc",
        "execute.cc",
        "execute_node.cc",
    ],
    hdrs = [
        "execute.h",
        "execute_node.h",
    ],
    copts = tf_copts(),
    deps = [
        ":context",
        ":copy_to_device_node",
        ":eager_executor",
        ":eager_op_rewrite_registry",
        ":eager_operation",
        ":kernel_and_device",
        ":placement_utils",
        ":tensor_handle",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/memory",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/compiler/jit:common",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:scoped_memory_debug_annotation",
        "//tensorflow/c/eager:abstract_function",
        "//tensorflow/core/platform:errors",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core/distributed_runtime/eager:remote_mgr",
            "//tensorflow/core:core_cpu_lib",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/distributed_runtime/eager:eager_client",
            "//tensorflow/core/distributed_runtime/eager:remote_execute_node",
            "//tensorflow/core/distributed_runtime/eager:remote_copy_node",
        ],
    }),
)

tf_cuda_library(
    name = "placement_utils",
    srcs = [
        "placement_utils.cc",
    ],
    hdrs = [
        "placement_utils.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":custom_device",
        ":attr_builder",
        ":eager_operation",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:core_cpu_lib",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
        ],
    }),
)

tf_cuda_library(
    name = "attr_builder",
    srcs = ["attr_builder.cc"],
    hdrs = ["attr_builder.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        # Only the TF_AttrType enum is required, so pull in just the C headers.
        # TODO(b/113535673): Break this dependency and avoid the C header completely.
        "//tensorflow/c:tf_attrtype",
        "@farmhash_archive//:farmhash",
        "//tensorflow/c/eager:abstract_op_attrs",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:core_cpu_internal",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
        ],
    }),
)

tf_cc_test(
    name = "attr_builder_test",
    srcs = ["attr_builder_test.cc"],
    deps = [
        ":attr_builder",
        "//tensorflow/c:c_api",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:types_proto_cc",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "attr_builder.h",
        "context.h",
        "custom_device.h",
        "custom_device_op_handler.h",
        "eager_executor.h",
        "eager_operation.h",
        "kernel_and_device.h",
        "tensor_handle.h",
        "tensor_handle_data.h",
    ],
    visibility = [
        "//tensorflow/core/function:__pkg__",
        "//tensorflow/python:__subpackages__",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = [
            "*_test.cc",
            "remote_*",
        ],
    ),
)
