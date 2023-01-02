# Description:
# Package op defines functions for adding TensorFlow operations to a Graph.

# gazelle:exclude wrappers.go

load("//tensorflow:tensorflow.bzl", "filegroup", "genrule")
load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

package(default_visibility = ["//tensorflow/go:__subpackages__"])

licenses(["notice"])  # Apache 2.0

go_library(
    name = "op",
    srcs = [
        "generate.go",
        "gradients.go",
        "op.go",
        "scope.go",
        ":wrappers",  # keep
    ],
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/op",
    deps = ["//tensorflow/go"],
)

go_test(
    name = "op_test",
    size = "small",
    srcs = [
        "gradients_test.go",
        "op_test.go",
        "scope_test.go",
    ],
    embed = [":op"],
    deps = ["//tensorflow/go"],
)

genrule(
    name = "wrappers",
    outs = ["wrappers.go"],
    cmd = """$(location //tensorflow/go/genop:genop) \
             -outfile $(location wrappers.go) \
	     -api_def_dirs tensorflow/core/api_def/base_api""",
    tools = ["//tensorflow/go/genop"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//visibility:public"],
)
