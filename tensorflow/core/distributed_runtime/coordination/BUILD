load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.default.bzl", "filegroup", "tf_grpc_cc_dependencies")
load(
    "//tensorflow:tensorflow.bzl",
    "if_oss",
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
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf:coordination_config_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
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
        ":coordination_service_error_util",
        "//tensorflow/compiler/xla/pjrt/distributed:protocol_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/framework:device_attributes_proto_cc",
        "//tensorflow/core/protobuf:coordination_config_proto_cc",
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "coordination_service_test",
    srcs = ["coordination_service_test.cc"],
    tags = if_oss([
        "manual",
        "no_oss",
    ]),  # b/169705709, no protobuf matchers in OSS.
    deps = [
        ":coordination_client",
        ":coordination_service",
        ":coordination_service_impl",
        "//tensorflow/compiler/xla/pjrt/distributed:protocol_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/distributed_runtime:test_utils",
        "//tensorflow/core/framework:device_attributes_proto_cc",
        "//tensorflow/core/protobuf:coordination_config_proto_cc",
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tf_cuda_library(
    name = "coordination_service_agent",
    srcs = ["coordination_service_agent.cc"],
    hdrs = ["coordination_service_agent.h"],
    deps = [
        ":coordination_client",
        ":coordination_service_error_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/protobuf:coordination_config_proto_cc",
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "coordination_service_agent_test",
    srcs = ["coordination_service_agent_test.cc"],
    deps = [
        ":coordination_client",
        ":coordination_service_agent",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/protobuf:coordination_config_proto_cc",
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
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
        ":coordination_service_agent",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/time",
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
        "//tensorflow/core:lib",
        ":coordination_service",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "coordination_service_error_util",
    hdrs = ["coordination_service_error_util.h"],
    deps = [
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "coordination_service_barrier_proxy",
    srcs = ["coordination_service_barrier_proxy.cc"],
    hdrs = ["coordination_service_barrier_proxy.h"],
    deps = [
        ":coordination_service_agent",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "coordination_service_barrier_proxy_test",
    srcs = ["coordination_service_barrier_proxy_test.cc"],
    deps = [
        ":coordination_service_agent",
        ":coordination_service_barrier_proxy",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:coordination_config_proto_cc",
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "coordination_service_error_util_test",
    srcs = ["coordination_service_error_util_test.cc"],
    deps = [
        ":coordination_service_error_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
    ],
)

tf_cc_test(
    name = "coordination_service_recoverable_job_test",
    srcs = ["coordination_service_recoverable_job_test.cc"],
    deps = [
        ":coordination_client",
        ":coordination_service",
        ":coordination_service_agent",
        ":coordination_service_impl",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/distributed_runtime/rpc:async_service_interface",
        "//tensorflow/core/distributed_runtime/rpc/coordination:grpc_coordination_client",
        "//tensorflow/core/distributed_runtime/rpc/coordination:grpc_coordination_service_impl",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/protobuf:coordination_config_proto_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/tsl/platform:mutex",
    ] + tf_grpc_cc_dependencies(),
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "coordination_client.h",
        "coordination_service.h",
    ],
    visibility = [
        "//tensorflow/core/function:__pkg__",
        "//tensorflow/python:__subpackages__",
    ],
)
