package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

load("@protobuf//:protobuf.bzl", "py_proto_library")

exports_files(["pprof/LICENSE"])

py_proto_library(
    name = "pprof_proto_py",
    srcs = ["proto/profile.proto"],
    default_runtime = "@protobuf//:protobuf_python",
    protoc = "@protobuf//:protoc",
    srcs_version = "PY2AND3",
    deps = ["@protobuf//:protobuf_python"],
)
