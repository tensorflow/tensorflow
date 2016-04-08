# Description:
# A distributed runtime for TensorFlow, which allows graph execution
# to be distributed and performed in parallel across multiple
# processes.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

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

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

load("//tensorflow:tensorflow.bzl", "tf_cc_tests")

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
)
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(default_visibility = [
    "//tensorflow:internal",
])

cc_library(
    name = "worker_env",
    hdrs = ["worker_env.h"],
    deps = [],
)

cc_library(
    name = "worker_interface",
    hdrs = ["worker_interface.h"],
    deps = [
        ":call_options",
        "//tensorflow/core:lib",
        "//tensorflow/core:worker_proto_cc",
    ],
)

cc_library(
    name = "call_options",
    srcs = ["call_options.cc"],
    hdrs = ["call_options.h"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "call_options_test",
    size = "small",
    srcs = ["call_options_test.cc"],
    deps = [
        ":call_options",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "worker_cache",
    hdrs = ["worker_cache.h"],
    deps = ["//tensorflow/core:protos_all_cc"],
)

cc_library(
    name = "remote_device",
    srcs = ["remote_device.cc"],
    hdrs = ["remote_device.h"],
    deps = [
        ":process_util",
        ":worker_cache",
        ":worker_interface",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:worker_proto_cc",
    ],
)

cc_library(
    name = "master_interface",
    hdrs = ["master_interface.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:master_proto_cc",
    ],
)

cc_library(
    name = "master",
    srcs = ["master.cc"],
    hdrs = ["master.h"],
    deps = [
        ":call_options",
        ":master_env",
        ":master_session_interface",
        ":process_util",
        ":remote_device",
        ":worker_cache",
        ":worker_interface",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:worker_proto_cc",
    ],
)

cc_library(
    name = "master_session",
    srcs = ["master_session.cc"],
    hdrs = ["master_session.h"],
    deps = [
        ":master_env",
        ":master_session_interface",
        ":process_util",
        ":simple_graph_execution_state",
        ":worker_cache",
        ":worker_interface",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "build_graph_options",
    srcs = ["build_graph_options.cc"],
    hdrs = ["build_graph_options.h"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "simple_graph_execution_state",
    srcs = ["simple_graph_execution_state.cc"],
    hdrs = ["simple_graph_execution_state.h"],
    deps = [
        ":build_graph_options",
        ":process_util",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:worker_proto_cc",
    ],
)

cc_library(
    name = "rendezvous_mgr_interface",
    srcs = [],
    hdrs = ["rendezvous_mgr_interface.h"],
    deps = [
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "master_session_interface",
    srcs = [],
    hdrs = ["master_session_interface.h"],
    deps = ["//tensorflow/core:lib"],
)

cc_library(
    name = "base_rendezvous_mgr",
    srcs = ["base_rendezvous_mgr.cc"],
    hdrs = ["base_rendezvous_mgr.h"],
    deps = [
        ":process_util",
        ":rendezvous_mgr_interface",
        ":worker_cache",
        ":worker_env",
        ":worker_interface",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "master_env",
    hdrs = ["master_env.h"],
)

cc_library(
    name = "graph_mgr",
    srcs = ["graph_mgr.cc"],
    hdrs = ["graph_mgr.h"],
    deps = [
        ":process_util",
        ":rendezvous_mgr_interface",
        ":worker_env",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:worker_proto_cc",
    ],
)

cc_library(
    name = "process_util",
    srcs = ["process_util.cc"],
    hdrs = ["process_util.h"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "worker_cache_partial",
    srcs = ["worker_cache_partial.cc"],
    hdrs = ["worker_cache_partial.h"],
    deps = [
        ":process_util",
        ":worker_cache",
        ":worker_interface",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:worker_proto_cc",
    ],
)

cc_library(
    name = "worker_cache_logger",
    srcs = ["worker_cache_logger.cc"],
    hdrs = ["worker_cache_logger.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "server_lib",
    srcs = ["server_lib.cc"],
    hdrs = ["server_lib.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

# TODO(mrry): Move executor_test.cc to ../common_runtime when once it no longer depends
# on grpc_testlib.
tf_cc_tests(
    size = "medium",
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    tests = [
        "executor_test.cc",
        #"master_test.cc",  # TODO(b/27683709): Re-enable when not flaky.
    ],
    deps = [
        "@grpc//:grpc++_unsecure",
        ":master",
        ":process_util",
        ":remote_device",
        ":worker_interface",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:master_service_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_testlib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_cache",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:dense_update_ops",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:variable_ops",
    ],
)

tf_cc_tests(
    size = "small",
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    tests = [
        "remote_device_test.cc",
    ],
    deps = [
        "@grpc//:grpc++_unsecure",
        ":master",
        ":process_util",
        ":remote_device",
        ":worker_interface",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:master_service_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_testlib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_cache",
    ],
)
