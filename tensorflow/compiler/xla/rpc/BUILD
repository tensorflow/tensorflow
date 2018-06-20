licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load(
    "//tensorflow/compiler/xla:xla.bzl",
    "xla_proto_library",
    "xla_py_grpc_library",
)

xla_proto_library(
    name = "xla_service_proto",
    srcs = ["xla_service.proto"],
    use_grpc_plugin = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla:xla_proto",
    ],
)

cc_library(
    name = "grpc_stub",
    srcs = ["grpc_stub.cc"],
    hdrs = ["grpc_stub.h"],
    deps = [
        ":xla_service_proto",
        "//tensorflow/compiler/xla:service_interface",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ],
)

tf_cc_binary(
    name = "grpc_service_main_cpu",
    srcs = ["grpc_service_main.cc"],
    deps = [
        ":grpc_service",
        "//tensorflow:grpc++",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "grpc_client_test",
    srcs = ["grpc_client_test.cc"],
    data = [
        "//tensorflow/compiler/xla/rpc:grpc_service_main_cpu",
    ],
    deps = [
        ":grpc_stub",
        "//tensorflow:grpc++",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client/xla_client:xla_builder",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "grpc_service",
    srcs = ["grpc_service.cc"],
    hdrs = ["grpc_service.h"],
    deps = [
        ":xla_service_proto",
        "//tensorflow:grpc++",
        "//tensorflow/compiler/xla/service",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ],
)
