package(default_visibility = [
    "//tensorflow:internal",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

cc_library(
    name = "remote_tensor_handle",
    hdrs = ["remote_tensor_handle.h"],
    deps = [
        "//tensorflow/core:eager_service_proto_cc",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "eager_client",
    hdrs = ["eager_client.h"],
    deps = [
        "//tensorflow/core:eager_service_proto_cc",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "remote_execute_node",
    hdrs = ["remote_execute_node.h"],
    deps = [
        ":eager_client",
        "//tensorflow/core:eager_service_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime/eager:eager_executor",
    ],
)

cc_library(
    name = "eager_service_impl",
    srcs = ["eager_service_impl.cc"],
    hdrs = [
        "eager_service_impl.h",
    ],
    deps = [
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:eager_service_proto_cc",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/common_runtime/eager:execute",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime:worker_cache",
        "//tensorflow/core/distributed_runtime:worker_cache_wrapper",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime/eager:remote_tensor_handle",
        "//tensorflow/core/distributed_runtime/rpc:rpc_rendezvous_mgr",
        "@grpc",
        "@grpc//:grpc++",
    ],
)

tf_cc_test(
    name = "eager_service_impl_test",
    srcs = ["eager_service_impl_test.cc"],
    deps = [
        ":eager_service_impl",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/core:eager_service_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime/rpc:rpc_rendezvous_mgr",
    ],
)
