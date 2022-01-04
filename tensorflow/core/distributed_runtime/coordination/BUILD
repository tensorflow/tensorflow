load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "filegroup")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependencies")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cuda_library",
)

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "coordination_client",
    hdrs = ["coordination_client.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
    ],
)

cc_library(
    name = "coordination_service",
    hdrs = ["coordination_service.h"],
    deps = [
        ":coordination_client",
        "//tensorflow/core:lib",
    ],
)

# Keeping the implementation as a separate build target.
# This is an alwayslink library for statically registering "standalone" implementation.
# Other implementations of the service will be provided in the future.
tf_cuda_library(
    name = "coordination_service_impl",
    srcs = ["coordination_service.cc"],
    deps = [
        ":coordination_client",
        ":coordination_service",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:device_mgr",
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "coordination_service_test",
    srcs = ["coordination_service_test.cc"],
    deps = [
        ":coordination_client",
        ":coordination_service",
        ":coordination_service_impl",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:device_factory",
        "//tensorflow/core/common_runtime:device_mgr",
        "//tensorflow/core/distributed_runtime:rpc_collective_executor_mgr",
        "//tensorflow/core/distributed_runtime:session_mgr",
        "//tensorflow/core/distributed_runtime:test_utils",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cuda_library(
    name = "coordination_service_agent",
    srcs = ["coordination_service_agent.cc"],
    hdrs = ["coordination_service_agent.h"],
    deps = [
        ":coordination_client",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:device_mgr",
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "coordination_service_rpc_handler",
    srcs = ["coordination_service_rpc_handler.cc"],
    hdrs = [
        "coordination_service_rpc_handler.h",
    ],
    deps = [
        ":coordination_service_agent",
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        ":coordination_service",
    ] + tf_grpc_cc_dependencies(),
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "coordination_client.h",
        "coordination_service.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__subpackages__",
    ],
)
