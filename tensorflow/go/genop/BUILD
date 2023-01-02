# Description:
# Generation of Go source files for TensorFlow ops.

# gazelle:exclude generate.go

load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

package(default_visibility = ["//tensorflow/go:__subpackages__"])

licenses(["notice"])  # Apache 2.0

go_library(
    name = "genop_lib",
    srcs = ["main.go"],
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/genop",
    deps = ["//tensorflow/go/genop/internal"],
)

go_binary(
    name = "genop",
    embed = [":genop_lib"],
)
