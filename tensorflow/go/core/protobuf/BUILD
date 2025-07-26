# Author: KleaSCM
# Email: KleaSCM@gmail.com
# File: tensorflow/go/core/protobuf/BUILD
# Description: Go protobuf build target for TensorFlow core protos. This target enables the Go API to use
# type-safe, maintainable configuration objects (e.g., ConfigProto, GPUOptions) for session configuration.
#
# This file connects the generated Go protobuf code to the rest of the Go API, making it possible to
# use native Go types for TensorFlow session options and other core configuration features.

load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")

go_proto_library(
    name = "for_core_protos_go_proto",
    compilers = ["@io_bazel_rules_go//proto:go_proto"],
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto",
    proto = "//tensorflow/core/protobuf:for_core_protos",
    visibility = ["//visibility:public"],
)
