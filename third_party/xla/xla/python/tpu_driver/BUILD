load(
    "//xla/python/tpu_driver:platform/external/tools.bzl",
    "external_deps",
    "go_grpc_library",
)
load("@local_tsl//tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
load("@local_tsl//tsl/platform:build_config.bzl", "tf_proto_library")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "tpu_driver_proto",
    srcs = ["tpu_driver.proto"],
    cc_api_version = 2,
    protodeps = [],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "tpu_service_proto",
    srcs = ["tpu_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    create_grpc_library = True,
    protodeps = [
        ":tpu_driver_proto",
        "//xla:xla_data_proto",
        "//xla:xla_proto",
        "//xla/service:hlo_proto",
    ],
    use_grpc_namespace = True,
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_driver_proto_cc",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/service:hlo_proto_cc",
        "@local_tsl//tsl/platform:logging",
    ] + external_deps(),
)

cc_library(
    name = "grpc_tpu_driver",
    srcs = [
        "grpc_tpu_driver.cc",
    ],
    hdrs = ["grpc_tpu_driver.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_driver",
        ":tpu_driver_proto_cc",
        ":tpu_service_cc_grpc_proto",
        ":tpu_service_proto_cc",
        "//xla:status",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:hlo_proto_cc",
        "@local_tsl//tsl/platform:logging",
    ] + tsl_grpc_cc_dependencies() + external_deps(),
    alwayslink = 1,
)

cc_library(
    name = "recording_tpu_driver",
    srcs = [
        "recording_tpu_driver.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_driver",
        ":tpu_driver_proto_cc",
        ":tpu_service_cc_grpc_proto",
        ":tpu_service_proto_cc",
        "//xla:status",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/base",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
    ] + external_deps(),
    alwayslink = 1,
)

cc_library(
    name = "pod_tpu_driver",
    srcs = ["pod_tpu_driver.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":grpc_tpu_driver",
        ":tpu_driver",
        ":tpu_driver_proto_cc",
        "//xla/pjrt:semaphore",
        "//xla/pjrt:worker_thread",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ] + tsl_grpc_cc_dependencies() + external_deps(),
    alwayslink = 1,
)

go_grpc_library(
    name = "tpu_service_go_grpc",
    srcs = [":tpu_service_proto"],
    compatible_with = ["//buildenv/target:gce"],
    deps = [":tpu_service_go_proto"],
)
