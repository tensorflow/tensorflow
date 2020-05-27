load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library_cc")
load("//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependency")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

licenses(["notice"])

package(default_visibility = ["//tensorflow:internal"])

tf_proto_library_cc(
    name = "protocol_proto",
    srcs = ["protocol.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    use_grpc_namespace = True,
)

cc_library(
    name = "protocol",
    hdrs = ["protocol.h"],
)

cc_library(
    name = "key_value_store",
    srcs = ["key_value_store.cc"],
    hdrs = ["key_value_store.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        tf_grpc_cc_dependency(),
    ],
)

cc_library(
    name = "service",
    srcs = ["service.cc"],
    hdrs = ["service.h"],
    deps = [
        ":key_value_store",
        ":protocol",
        ":protocol_proto_cc",
        ":util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "service_test",
    srcs = ["service_test.cc"],
    deps = [
        ":protocol_proto_cc",
        ":service",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "client",
    srcs = [
        "client.cc",
    ],
    hdrs = [
        "client.h",
    ],
    deps = [
        ":protocol",
        ":protocol_proto_cc",
        ":util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        tf_grpc_cc_dependency(),
    ],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/compiler/xla:status",
        tf_grpc_cc_dependency(),
    ],
)

cc_library(
    name = "distributed",
    srcs = ["distributed.cc"],
    hdrs = ["distributed.h"],
    deps = [
        ":client",
        ":service",
        "//tensorflow/compiler/xla:statusor",
        tf_grpc_cc_dependency(),
    ],
)

tf_cc_test(
    name = "client_server_test",
    srcs = ["client_server_test.cc"],
    deps = [
        ":client",
        ":protocol_proto_cc",
        ":service",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/time",
        tf_grpc_cc_dependency(),
    ],
)
