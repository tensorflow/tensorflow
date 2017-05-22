# Description:
# TensorFlow Debugger (tfdbg).
#
# Public target(s):
#
# ":debug" - Depending on this target causes a concrete implementation of
#    DebuggerState to be constructed at initialization time, enabling
#    TensorFlow Debugger (tfdbg) support. For details, please see
#    core/common_runtime/debugger_state_interface.h.

package(
    default_visibility = ["//tensorflow:internal"],
    features = ["-parse_headers"],
)

licenses(["notice"])  # Apache 2.0

# Google-internal rules omitted.
load(
    "//tensorflow:tensorflow.bzl",
    "check_deps",
    "tf_copts",
    "tf_cc_test",
    "tf_cuda_library",
)

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
    "tf_proto_library_cc",
)
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)

# Check that tensorflow/core:tensorflow does not depend on grpc.
check_deps(
    name = "core_tensorflow_check_deps",
    disallowed_deps = ["@grpc//:grpc++_unsecure"],
    deps = ["//tensorflow/core:tensorflow"],
)

tf_proto_library_cc(
    name = "debug_service_proto",
    srcs = ["debug_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    protodeps = ["//tensorflow/core:protos_all"],
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

tf_cuda_library(
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
        "//tensorflow/core:gpu_tracer",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "debugger_state_impl",
    srcs = ["debugger_state_impl.cc"],
    hdrs = ["debugger_state_impl.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        ":debug_graph_utils",
        ":debug_io_utils",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "debug_graph_utils",
    srcs = ["debug_graph_utils.cc"],
    hdrs = ["debug_graph_utils.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "debug_io_utils",
    srcs = ["debug_io_utils.cc"],
    hdrs = ["debug_io_utils.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        ":debug_service_proto_cc",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
        "@grpc//:grpc++_unsecure",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "debug_grpc_testlib",
    srcs = ["debug_grpc_testlib.cc"],
    hdrs = ["debug_grpc_testlib.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        ":debug_graph_utils",
        ":debug_io_utils",
        ":debug_service_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@grpc//:grpc++_unsecure",
    ],
    alwayslink = 1,
)

# TODO(cais): Fix flakiness on GPU and change this back to a tf_cc_test_gpu.
#   See b/34081273.
tf_cc_test(
    name = "debug_gateway_test",
    size = "small",
    srcs = ["debug_gateway_test.cc"],
    args = ["--heap_check=local"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = ["no_gpu"],
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
        ":debug_grpc_testlib",
        ":debug_io_utils",
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
    tags = ["nomac"],  # b/38276817
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

# -----------------------------------------------------------------------------
# Google-internal targets.  These must be at the end for syncrepo.

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
