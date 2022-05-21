load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependencies")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "grpc_eager_service",
    srcs = ["grpc_eager_service.h"],
    hdrs = ["grpc_eager_service.h"],
    deps = [
        "//tensorflow/core/protobuf:eager_service_proto_cc",
        "//tensorflow/stream_executor/platform",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_eager_client",
    srcs = ["grpc_eager_client.cc"],
    hdrs = ["grpc_eager_client.h"],
    deps = [
        ":grpc_eager_service",
        "//tensorflow/core/platform:error_payloads",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/distributed_runtime/eager:eager_client",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_client_cq_tag",
        "//tensorflow/core/distributed_runtime/rpc:grpc_state",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/protobuf:eager_service_proto_cc",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_eager_service_impl",
    srcs = ["grpc_eager_service_impl.cc"],
    hdrs = ["grpc_eager_service_impl.h"],
    deps = [
        ":grpc_eager_service",
        "//tensorflow/core:framework",
        "//tensorflow/core:ptr_util",
        "//tensorflow/core/distributed_runtime/eager:eager_service_impl",
        "//tensorflow/core/distributed_runtime/rpc:async_service_interface",
        "//tensorflow/core/distributed_runtime/rpc:grpc_call",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_cache",
        "//tensorflow/core/protobuf:eager_service_proto_cc",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "grpc_eager_client_test",
    size = "small",
    srcs = [
        "grpc_eager_client_test.cc",
    ],
    deps = [
        ":grpc_eager_client",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/platform:blocking_counter",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:strcat",
    ],
)
