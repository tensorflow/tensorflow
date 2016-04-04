# Description:
#   RPC communication interfaces and implementations for TensorFlow.

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

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cuda_library",
    "tf_cc_tests",
)

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
    name = "grpc_util",
    srcs = [],
    hdrs = ["grpc_util.h"],
    deps = [
        "@grpc//:grpc++_unsecure",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "grpc_client_cq_tag",
    srcs = [],
    hdrs = ["grpc_client_cq_tag.h"],
    deps = [
        "@grpc//:grpc++_unsecure",
        ":grpc_util",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "grpc_remote_worker",
    srcs = ["grpc_remote_worker.cc"],
    hdrs = ["grpc_remote_worker.h"],
    deps = [
        "@grpc//:grpc++_unsecure",
        ":grpc_client_cq_tag",
        ":grpc_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:worker_proto_cc",
        "//tensorflow/core:worker_service_proto_cc",
        "//tensorflow/core/distributed_runtime:process_util",
        "//tensorflow/core/distributed_runtime:worker_cache_logger",
        "//tensorflow/core/distributed_runtime:worker_interface",
    ],
)

cc_library(
    name = "grpc_channel",
    srcs = ["grpc_channel.cc"],
    hdrs = ["grpc_channel.h"],
    deps = [
        "@grpc//:grpc++_unsecure",
        ":grpc_util",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "grpc_call",
    srcs = [],
    hdrs = ["grpc_call.h"],
    deps = [
        "@grpc//:grpc++_unsecure",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "async_service_interface",
    srcs = [],
    hdrs = ["async_service_interface.h"],
    deps = [],
)

cc_library(
    name = "grpc_worker_cache",
    srcs = ["grpc_worker_cache.cc"],
    hdrs = ["grpc_worker_cache.h"],
    deps = [
        ":grpc_channel",
        ":grpc_client_cq_tag",
        ":grpc_remote_worker",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:worker_cache",
        "//tensorflow/core/distributed_runtime:worker_cache_logger",
        "//tensorflow/core/distributed_runtime:worker_cache_partial",
        "//tensorflow/core/distributed_runtime:worker_interface",
    ],
)

cc_library(
    name = "grpc_worker_service",
    srcs = ["grpc_worker_service.cc"],
    hdrs = ["grpc_worker_service.h"],
    deps = [
        "@grpc//:grpc++_unsecure",
        ":async_service_interface",
        ":grpc_call",
        ":grpc_util",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_lib",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:worker_proto_cc",
        "//tensorflow/core:worker_service_proto_cc",
        "//tensorflow/core/distributed_runtime:graph_mgr",
        "//tensorflow/core/distributed_runtime:process_util",
        "//tensorflow/core/distributed_runtime:rendezvous_mgr_interface",
        "//tensorflow/core/distributed_runtime:worker_cache",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime:worker_interface",
    ],
)

cc_library(
    name = "grpc_remote_master",
    srcs = ["grpc_remote_master.cc"],
    hdrs = ["grpc_remote_master.h"],
    deps = [
        "@grpc//:grpc++_unsecure",
        ":grpc_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:master_service_proto_cc",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/distributed_runtime:master_interface",
    ],
    alwayslink = 1,
)

cc_library(
    name = "grpc_master_service",
    srcs = ["grpc_master_service.cc"],
    hdrs = ["grpc_master_service.h"],
    deps = [
        "@grpc//:grpc++_unsecure",
        ":async_service_interface",
        ":grpc_call",
        ":grpc_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:master_service_proto_cc",
        "//tensorflow/core/distributed_runtime:master",
        "//tensorflow/core/distributed_runtime:master_interface",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rpc_rendezvous_mgr",
    srcs = ["rpc_rendezvous_mgr.cc"],
    hdrs = ["rpc_rendezvous_mgr.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:base_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime:process_util",
        "//tensorflow/core/distributed_runtime:worker_cache",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime:worker_interface",
    ],
)

cc_library(
    name = "grpc_server_lib",
    srcs = ["grpc_server_lib.cc"],
    hdrs = ["grpc_server_lib.h"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
    deps = [
        "@grpc//:grpc++_unsecure",
        ":async_service_interface",
        ":grpc_channel",
        ":grpc_master_service",
        ":grpc_worker_cache",
        ":grpc_worker_service",
        ":rpc_rendezvous_mgr",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/distributed_runtime:graph_mgr",
        "//tensorflow/core/distributed_runtime:master_env",
        "//tensorflow/core/distributed_runtime:master_session",
        "//tensorflow/core/distributed_runtime:process_util",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime:worker_env",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "grpc_tensorflow_server",
    srcs = [
        "grpc_tensorflow_server.cc",
    ],
    deps = [
        "@grpc//:grpc++_unsecure",
        ":grpc_server_lib",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/distributed_runtime:server_lib",
    ],
)

tf_cuda_library(
    name = "grpc_testlib_ops",
    testonly = 1,
    srcs = ["grpc_testlib_ops.cc"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "grpc_testlib_server",
    testonly = 1,
    srcs = [
        "grpc_testlib_server.cc",
    ],
    deps = [
        "@grpc//:grpc++_unsecure",
        ":grpc_server_lib",
        ":grpc_testlib_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:dense_update_ops",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:matmul_op",
        "//tensorflow/core/kernels:reduction_ops",
        "//tensorflow/core/kernels:variable_ops",
    ],
)

tf_cuda_library(
    name = "grpc_testlib",
    testonly = 1,
    srcs = ["grpc_testlib.cc"],
    hdrs = ["grpc_testlib.h"],
    data = [
        ":grpc_testlib_server",
    ],
    deps = [
        ":grpc_session",
        ":grpc_testlib_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
    ],
    alwayslink = 1,
)

cc_library(
    name = "grpc_session",
    srcs = ["grpc_session.cc"],
    hdrs = ["grpc_session.h"],
    deps = [
        ":grpc_channel",
        ":grpc_remote_master",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/distributed_runtime:master_interface",
    ],
    alwayslink = 1,
)

tf_cc_tests(
    size = "small",
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + ["exclusive"],
    tests = [
        "grpc_channel_test.cc",
        "rpc_rendezvous_mgr_test.cc",
    ],
    deps = [
        ":grpc_channel",
        ":grpc_server_lib",
        ":grpc_session",
        ":grpc_testlib",
        ":rpc_rendezvous_mgr",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/distributed_runtime:process_util",
        "//tensorflow/core/distributed_runtime:server_lib",
    ],
)

tf_cc_tests(
    size = "medium",
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + ["exclusive"],
    tests = [
        "grpc_session_test.cc",
    ],
    deps = [
        ":grpc_channel",
        ":grpc_server_lib",
        ":grpc_session",
        ":grpc_testlib",
        ":rpc_rendezvous_mgr",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/distributed_runtime:process_util",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:dense_update_ops",
        "//tensorflow/core/kernels:matmul_op",
        "//tensorflow/core/kernels:variable_ops",
    ],
)
