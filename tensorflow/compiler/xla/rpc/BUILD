load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load(
    "//tensorflow/compiler/xla:xla.bzl",
    "xla_cc_binary",
    "xla_py_grpc_library",
)
load(
    "//tensorflow/tsl:tsl.bzl",
    "if_libtpu",
)
load(
    "//tensorflow/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "xla_service_proto",
    srcs = ["xla_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    create_grpc_library = True,
    create_service = True,
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
        ":xla_service_cc_grpc_proto",
        ":xla_service_proto_cc",
        "//tensorflow/compiler/xla:service_interface",
        "//tensorflow/compiler/xla:xla_data_cc_grpc_proto",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/tsl/distributed_runtime/rpc:grpc_util",
    ],
)

cc_library(
    name = "grpc_service_main_library",
    srcs = ["grpc_service_main.cc"],
    deps = [
        ":grpc_service",
        "@com_google_absl//absl/strings:str_format",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/util:command_line_flags",
    ] + tsl_grpc_cc_dependencies() + if_libtpu(
        if_false = ["//tensorflow/compiler/xla/service:cpu_plugin"],
        if_true = [],
    ),
)

xla_cc_binary(
    name = "grpc_service_main_cpu",
    deps = [
        ":grpc_service_main_library",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/tsl/protobuf:dnn_proto_cc_impl",
    ],
)

tf_cc_test(
    name = "grpc_client_test",
    srcs = ["grpc_client_test.cc"],
    data = [
        ":grpc_service_main_cpu",
    ],
    deps = [
        ":grpc_stub",
        "@com_google_absl//absl/strings:str_format",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:net",
        "//tensorflow/tsl/platform:subprocess",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/platform:path",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_service",
    srcs = ["grpc_service.cc"],
    hdrs = ["grpc_service.h"],
    deps = [
        ":xla_service_proto_cc",
        ":xla_service_cc_grpc_proto",
        "//tensorflow/compiler/xla/service",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/tsl/distributed_runtime/rpc:grpc_util",
    ] + tsl_grpc_cc_dependencies(),
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "xla_py_pb2",
#     has_services = True,
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":xla_service_proto"],
# )
#
# xla_py_grpc_library(
#     name = "xla_py_pb2_grpc",
#     srcs = [":xla_service_proto"],
#     deps = [":xla_py_pb2"],
# )
# copybara:uncomment_end
