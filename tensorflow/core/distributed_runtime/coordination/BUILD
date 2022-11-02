load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.default.bzl", "filegroup", "tf_grpc_cc_dependencies")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
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
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_client",
    ],
)

cc_library(
    name = "coordination_service_rpc_handler",
    srcs = ["coordination_service_rpc_handler.cc"],
    hdrs = [
        "coordination_service_rpc_handler.h",
    ],
    deps = [
        ":coordination_service_error_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/time",
        "//tensorflow/core:lib",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service",
        "//tensorflow/tsl/protobuf:coordination_service_proto_cc",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_agent",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "coordination_service_error_util",
    hdrs = ["coordination_service_error_util.h"],
    deps = [
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_error_util",
    ],
)

cc_library(
    name = "coordination_service_barrier_proxy",
    srcs = ["coordination_service_barrier_proxy.cc"],
    hdrs = ["coordination_service_barrier_proxy.h"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_agent",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "coordination_service_barrier_proxy_test",
    srcs = ["coordination_service_barrier_proxy_test.cc"],
    deps = [
        ":coordination_service_barrier_proxy",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/tsl/distributed_runtime:call_options",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_client",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_agent",
        "//tensorflow/tsl/protobuf:coordination_config_proto_cc",
        "//tensorflow/tsl/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "coordination_service_recoverable_job_test",
    srcs = ["coordination_service_recoverable_job_test.cc"],
    deps = [
        ":coordination_client",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/distributed_runtime/rpc/coordination:grpc_coordination_client",
        "//tensorflow/core/distributed_runtime/rpc/coordination:grpc_coordination_service_impl",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_agent",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_impl",
        "//tensorflow/tsl/distributed_runtime/rpc:async_service_interface",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/protobuf:coordination_config_proto_cc",
    ] + tf_grpc_cc_dependencies(),
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "coordination_client.h",
    ],
    visibility = [
        "//tensorflow/core/function:__pkg__",
        "//tensorflow/python:__subpackages__",
    ],
)
