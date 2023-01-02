# Description:
# Protocol buffer generation for tensorflow/tsl/profiler/protobuf.

load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")

package(default_visibility = ["//tensorflow/go:__subpackages__"])

licenses(["notice"])  # Apache 2.0

go_proto_library(
    name = "for_profiler_protos_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/tsl/profiler/protobuf/for_profiler_protos_go_proto",
    protos = [
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto",
    ],
    deps = [],
)
