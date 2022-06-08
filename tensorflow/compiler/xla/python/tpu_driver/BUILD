load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependencies")
load(
    "//tensorflow/compiler/xla/python/tpu_driver:platform/external/tools.bzl",
    "external_deps",
    "go_grpc_library",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

tf_proto_library(
    name = "tpu_driver_proto",
    srcs = ["tpu_driver.proto"],
    cc_api_version = 2,
    protodeps = [],
)

tf_proto_library(
    name = "tpu_service_proto",
    srcs = ["tpu_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    create_grpc_library = True,
    protodeps = [
        ":tpu_driver_proto",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo_proto",
    ],
    use_grpc_namespace = True,
)

cc_library(
    name = "tpu_driver",
    srcs = [
        "tpu_driver.cc",
    ],
    hdrs = [
        "event_id.h",
        "platform/external/compat.h",
        "tpu_driver.h",
    ],
    deps = [
        "@com_google_absl//absl/types:optional",
        "//tensorflow/core/platform:logging",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        ":tpu_driver_proto_cc",
    ] + external_deps(),
)

cc_library(
    name = "grpc_tpu_driver",
    srcs = [
        "grpc_tpu_driver.cc",
    ],
    hdrs = ["grpc_tpu_driver.h"],
    deps = [
        ":tpu_driver",
        "//tensorflow/core/platform:logging",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        ":tpu_service_proto_cc",
        ":tpu_service_cc_grpc_proto",
        ":tpu_driver_proto_cc",
    ] + tf_grpc_cc_dependencies() + external_deps(),
    alwayslink = 1,
)

cc_library(
    name = "direct_tpu_driver",
    srcs = ["direct_tpu_driver.cc"],
    compatible_with = [],
    deps = [
        ":tpu_driver",
        "@com_google_absl//absl/strings:str_format",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core/platform:logging",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        ":tpu_service_proto_cc",
        ":tpu_driver_proto_cc",
        "//tensorflow/compiler/xla/python/tpu_driver/client:libtpu",
    ] + external_deps(),
    alwayslink = 1,
)

cc_library(
    name = "recording_tpu_driver",
    srcs = [
        "recording_tpu_driver.cc",
    ],
    deps = [
        ":tpu_driver",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/types:optional",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:logging",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        ":tpu_driver_proto_cc",
        ":tpu_service_proto_cc",
        ":tpu_service_cc_grpc_proto",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
    ] + external_deps(),
    alwayslink = 1,
)

cc_library(
    name = "pod_tpu_driver",
    srcs = ["pod_tpu_driver.cc"],
    deps = [
        ":grpc_tpu_driver",
        ":tpu_driver",
        ":tpu_driver_proto_cc",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "//tensorflow/compiler/xla/pjrt:semaphore",
        "//tensorflow/compiler/xla/pjrt:worker_thread",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ] + tf_grpc_cc_dependencies() + external_deps(),
    alwayslink = 1,
)

go_grpc_library(
    name = "tpu_service_go_grpc",
    srcs = [":tpu_service_proto"],
    compatible_with = ["//buildenv/target:gce"],
    deps = [":tpu_service_go_proto"],
)
