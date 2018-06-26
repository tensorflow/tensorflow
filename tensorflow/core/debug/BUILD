# Description:
# TensorFlow Debugger (tfdbg).
#
# Public target(s):
#
# ":debug" - Depending on this target causes a concrete implementation of
#    DebuggerState to be constructed at initialization time, enabling
#    TensorFlow Debugger (tfdbg) support. For details, please see
#    core/common_runtime/debugger_state_interface.h.
# ":debug_callback_registry" - Depending on this target exposes a global
#    callback registry that will be used to record any observed tensors matching
#    a watch state.
# ":debug_node_key" - Defines a struct used for tracking tensors.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "check_deps",
    "tf_copts",
    "tf_cc_test",
    "tf_gpu_library",
)

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
    "tf_proto_library",
    "tf_proto_library_cc",
    "tf_additional_all_protos",
)
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "tf_gpu_tests_tags",
)

# Check that tensorflow/core:tensorflow does not depend on grpc.
check_deps(
    name = "core_tensorflow_check_deps",
    disallowed_deps = ["@grpc//:grpc++"],
    deps = ["//tensorflow/core:tensorflow"],
)

tf_proto_library(
    name = "debug_service_proto",
    srcs = [
        "debug_service.proto",
    ],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    protodeps = [
        ":debugger_event_metadata_proto",
        "//tensorflow/core/profiler:protos_all",
    ] + tf_additional_all_protos(),
    visibility = ["//tensorflow:__subpackages__"],
)

tf_proto_library(
    name = "debugger_event_metadata_proto",
    srcs = ["debugger_event_metadata.proto"],
    cc_api_version = 2,
)

cc_library(
    name = "debug",
    srcs = ["debug.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":debugger_state_impl",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:debug_ops_op_lib",
    ],
    alwayslink = 1,
)

tf_gpu_library(
    name = "debug_gateway_internal",
    srcs = ["debug_gateway.cc"],
    hdrs = ["debug_gateway.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        ":debug",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_gpu_library(
    name = "debugger_state_impl",
    srcs = ["debugger_state_impl.cc"],
    hdrs = ["debugger_state_impl.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        ":debug_graph_utils",
        ":debug_io_utils",
        "//tensorflow/core:core_cpu_internal",
    ],
    alwayslink = 1,
)

tf_gpu_library(
    name = "debug_graph_utils",
    srcs = ["debug_graph_utils.cc"],
    hdrs = ["debug_graph_utils.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_gpu_library(
    name = "debug_io_utils",
    srcs = ["debug_io_utils.cc"],
    hdrs = ["debug_io_utils.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        ":debug_callback_registry",
        ":debug_node_key",
        ":debug_service_proto_cc",
        ":debugger_event_metadata_proto_cc",
        "//tensorflow:grpc++",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_gpu_library(
    name = "debug_grpc_testlib",
    srcs = ["debug_grpc_testlib.cc"],
    hdrs = ["debug_grpc_testlib.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        ":debug_graph_utils",
        ":debug_io_utils",
        ":debug_service_proto_cc",
        ":debugger_event_metadata_proto_cc",
        "//tensorflow:grpc++",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_gpu_library(
    name = "debug_node_key",
    srcs = ["debug_node_key.cc"],
    hdrs = ["debug_node_key.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

# TODO(cais): Fix flakiness on GPU and change this back to a tf_cc_test_gpu.
#   See b/34081273.
tf_cc_test(
    name = "debug_gateway_test",
    size = "small",
    srcs = ["debug_gateway_test.cc"],
    args = ["--heap_check=local"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = [
        "no_cuda_on_cpu_tap",
        "no_gpu",
    ],
    deps = [
        ":debug",
        ":debug_gateway_internal",
        ":debug_graph_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:direct_session_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:debug_ops",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test(
    name = "debug_io_utils_test",
    size = "small",
    srcs = ["debug_io_utils_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":debug_callback_registry",
        ":debug_grpc_testlib",
        ":debug_io_utils",
        ":debug_node_key",
        ":debug_service_proto_cc",
        ":debugger_event_metadata_proto_cc",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform/default/build_config:platformlib",
    ],
)

tf_cc_test(
    name = "debug_graph_utils_test",
    size = "small",
    srcs = ["debug_graph_utils_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":debug_graph_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "grpc_session_debug_test",
    size = "medium",
    srcs = ["grpc_session_debug_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = [
        "no_oss",  # b/62956105: port conflicts.
        "nomac",  # b/38276817
    ],
    deps = [
        ":debug_grpc_testlib",
        ":debug_io_utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
        "//tensorflow/core/distributed_runtime/rpc:grpc_testlib",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:matmul_op",
    ],
)

tf_cc_test(
    name = "debug_grpc_io_utils_test",
    size = "small",
    srcs = ["debug_grpc_io_utils_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = [
        "no_oss",  # b/73962011
    ],
    deps = [
        ":debug_graph_utils",
        ":debug_grpc_testlib",
        ":debug_io_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "debug_callback_registry",
    srcs = ["debug_callback_registry.cc"],
    hdrs = ["debug_callback_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":debug_node_key",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

# TODO(cais): Add the following back in when tfdbg is supported on Android.
# filegroup(
#     name = "android_srcs",
#     srcs = [
#         "debug_graph_utils.cc",
#         "debug_graph_utils.h",
#         "debug_io_utils.cc",
#         "debug_io_utils.h",
#     ],
#     visibility = ["//visibility:public"],
# )
