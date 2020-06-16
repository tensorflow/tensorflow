load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cc_test_mkl",
    "tf_copts",
    "tf_cuda_library",
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
    licenses = ["notice"],  # Apache 2.0
)

# TODO(b/152902651): Remove this file once all circular dependencies are resolved.
tf_cuda_library(
    name = "core",
    srcs = [
        "core.cc",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":context",
        ":eager_operation",
        ":execute",
        ":tensor_handle",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:tf_tensor_internal",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/c/eager:context_interface",
        "//tensorflow/c/eager:tensor_handle_interface",
        "//tensorflow/c/eager:operation_interface",
        "//tensorflow/core/distributed_runtime:rendezvous_mgr_interface",
        "//tensorflow/core/distributed_runtime:worker_env",
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
            "//tensorflow/core/distributed_runtime:device_resolver_distributed",
            "//tensorflow/core/distributed_runtime:rpc_collective_executor_mgr",
            "//tensorflow/core/distributed_runtime:worker_cache",
            "//tensorflow/core/distributed_runtime:server_lib",
            "//tensorflow/core/distributed_runtime:worker_session",
            "//tensorflow/core/distributed_runtime/eager:eager_client",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
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
        ":context",
        ":eager_executor",
        ":kernel_and_device",
        ":tensor_handle",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/c/eager:operation_interface",
        "//tensorflow/c/eager:tensor_handle_interface",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:platform_port",
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
            "//tensorflow/c/eager:tensor_handle_interface",
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
    "//tensorflow/core/profiler/lib:connected_traceme",
    "//tensorflow/core/profiler/lib:traceme",
    "//tensorflow/core/profiler/lib:traceme_encode",
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
        "//tensorflow:windows": KERNEL_AND_DEVICE_DEPS,
        "//conditions:default": KERNEL_AND_DEVICE_DEPS + [
            "//tensorflow/compiler/jit:xla_kernel_creator_util",
        ],
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
    ],
    deps = [
        ":context",
        ":copy_to_device_node",
        ":eager_executor",
        ":eager_op_rewrite_registry",
        ":eager_operation",
        ":kernel_and_device",
        ":tensor_handle",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/memory",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/compiler/jit:common",
        "//tensorflow/core/profiler/lib:traceme",
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

cc_library(
    name = "mkl_eager_op_rewrite",
    srcs = ["mkl_eager_op_rewrite.cc"],
    copts = tf_copts(allow_exceptions = True),
    deps = [
        ":eager_op_rewrite_registry",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:mkl_graph_util",
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
        ":mkl_eager_op_rewrite",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
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
    deps = [
        ":context",
        ":copy_to_device_node",
        ":eager_executor",
        ":eager_op_rewrite_registry",
        ":eager_operation",
        ":kernel_and_device",
        ":tensor_handle",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/memory",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/compiler/jit:common",
        "//tensorflow/core/profiler/lib:traceme",
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
    name = "attr_builder",
    srcs = ["attr_builder.cc"],
    hdrs = ["attr_builder.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        # Only the TF_AttrType enum is required, so pull in just the C headers.
        # TODO(b/113535673): Break this dependency and avoid the C header completely.
        "//tensorflow/c:tf_attrtype",
        "@farmhash_archive//:farmhash",
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
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "attr_builder.h",
        "context.h",
        "eager_executor.h",
        "eager_operation.h",
        "kernel_and_device.h",
        "tensor_handle.h",
        "tensor_handle_data.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__pkg__",
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
