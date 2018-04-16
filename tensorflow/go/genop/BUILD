load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    importpath = "github.com/tensorflow/tensorflow/go/genop",
    visibility = ["//visibility:private"],
    deps = ["//tensorflow/tensorflow/go/genop/internal:go_default_library"],
)

go_binary(
    name = "genop",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
)
