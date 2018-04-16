load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "api_def_map.go",
        "genop.go",
        "lib.go",
    ],
    cgo = True,
    clinkopts = ["-ltensorflow"],
    copts = ["-Itensorflow/go/genop/internal"],
    importpath = "github.com/tensorflow/tensorflow/go/genop/internal",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tensorflow/go/genop/internal/proto/tensorflow/core/framework_go_proto:go_default_library",
        "@com_github_golang_protobuf//proto:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["genop_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//tensorflow/tensorflow/go/genop/internal/proto/tensorflow/core/framework_go_proto:go_default_library",
        "@com_github_golang_protobuf//proto:go_default_library",
    ],
)
