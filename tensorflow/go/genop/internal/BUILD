load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

go_library(
    name = "internal",
    srcs = [
        "api_def_map.go",
        "genop.go",
        "lib.go",
    ],
    cdeps = [
        "//tensorflow/go:libtensorflow",
    ],  # keep
    cgo = True,
    copts = [],  # keep
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/genop/internal",
    visibility = ["//tensorflow/go/genop:__subpackages__"],
    deps = [
        "//tensorflow/go/core/framework:api_def_go_proto",
        "//tensorflow/go/core/framework:op_def_go_proto",
        "@org_golang_google_protobuf//encoding/prototext:go_default_library",
        "@org_golang_google_protobuf//proto:go_default_library",
    ],
)

go_test(
    name = "internal_test",
    size = "small",
    srcs = ["genop_test.go"],
    embed = [":internal"],
    deps = [
        "//tensorflow/go/core/framework:api_def_go_proto",
        "//tensorflow/go/core/framework:op_def_go_proto",
        "@org_golang_google_protobuf//encoding/prototext:go_default_library",
    ],
)
