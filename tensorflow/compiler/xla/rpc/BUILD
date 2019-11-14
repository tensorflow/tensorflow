load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load(
    "//tensorflow/core/platform:default/build_config.bzl",
    "tf_proto_library_cc",
)
load(
    "//tensorflow/compiler/xla:xla.bzl",
    "xla_py_grpc_library",
)

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

tf_proto_library_cc(
    name = "xla_service_proto",
    srcs = ["xla_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    protodeps = [
        "//tensorflow/compiler/xla:xla_proto",
    ],
    use_grpc_namespace = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "grpc_stub",
    srcs = ["grpc_stub.cc"],
    hdrs = ["grpc_stub.h"],
    deps = [
        ":xla_service_proto_cc",
        "//tensorflow/compiler/xla:service_interface",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ],
)

cc_library(
    name = "grpc_service_main_library",
    srcs = ["grpc_service_main.cc"],
    deps = [
        ":grpc_service",
        "//tensorflow:grpc++",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_binary(
    name = "grpc_service_main_cpu",
    deps = [
        ":grpc_service_main_library",
        "//tensorflow/compiler/xla/service:cpu_plugin",
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
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "grpc_service",
    srcs = ["grpc_service.cc"],
    hdrs = ["grpc_service.h"],
    deps = [
        ":xla_service_proto_cc",
        "//tensorflow:grpc++",
        "//tensorflow/compiler/xla/service",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ],
)
