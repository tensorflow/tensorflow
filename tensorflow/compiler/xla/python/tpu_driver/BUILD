load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library_cc")
load(
    "//tensorflow/compiler/xla/python/tpu_driver:platform/external/tools.bzl",
    "external_deps",
    "go_grpc_library",
    "go_proto_library",
)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

tf_proto_library_cc(
    name = "tpu_driver_proto",
    srcs = ["tpu_driver.proto"],
    cc_api_version = 2,
    protodeps = [],
)

tf_proto_library_cc(
    name = "tpu_service_proto",
    srcs = ["tpu_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
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
        "//tensorflow/compiler/xla/python/tpu_driver:tpu_driver_proto_cc",
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
        "//tensorflow:grpc++",
        "//tensorflow/core/platform:logging",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/python/tpu_driver:tpu_service_proto_cc",
        "//tensorflow/compiler/xla/python/tpu_driver:tpu_driver_proto_cc",
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
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/python/tpu_driver:tpu_driver_proto_cc",
        "//tensorflow/compiler/xla/python/tpu_driver:tpu_service_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
    ] + external_deps(),
    alwayslink = 1,
)

go_proto_library(
    name = "tpu_service_go_proto",
    compatible_with = ["//buildenv/target:gce"],
    deps = [":tpu_service_proto"],
)

go_grpc_library(
    name = "tpu_service_go_grpc",
    srcs = [":tpu_service_proto"],
    compatible_with = ["//buildenv/target:gce"],
    deps = [":tpu_service_go_proto"],
)
