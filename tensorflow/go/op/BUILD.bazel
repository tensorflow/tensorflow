load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "generate.go",
        "op.go",
        "scope.go",
        "wrappers.go",
    ],
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/op",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/go:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "op_test.go",
        "scope_test.go",
    ],
    embed = [":go_default_library"],
    deps = ["//tensorflow/go:go_default_library"],
)
