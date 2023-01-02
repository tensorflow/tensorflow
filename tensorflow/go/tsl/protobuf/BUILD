# Description:
# Protocol buffer generation for tensorflow/tsl/protobuf.

load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")

package(default_visibility = ["//tensorflow/go:__subpackages__"])

licenses(["notice"])  # Apache 2.0

go_proto_library(
    name = "for_core_protos_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/tsl/protobuf/for_core_protos_go_proto",
    protos = [
        "//tensorflow/tsl/protobuf:bfc_memory_map_proto",
        "//tensorflow/tsl/protobuf:coordination_config_proto",
        "//tensorflow/tsl/protobuf:distributed_runtime_payloads_proto",
        "//tensorflow/tsl/protobuf:error_codes_proto_impl",
        "//tensorflow/tsl/protobuf:histogram_proto",
        "//tensorflow/tsl/protobuf:rpc_options_proto",
        "//tensorflow/tsl/protobuf:test_log_proto",
    ],
    deps = [
        "@com_github_golang_protobuf//ptypes/any",
        "@com_github_golang_protobuf//ptypes/wrappers",
    ],
)
