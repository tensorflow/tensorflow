package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

load("@protobuf_archive//:protobuf.bzl", "py_proto_library")

exports_files(["pprof/LICENSE"])

py_proto_library(
    name = "pprof_proto_py",
    srcs = ["proto/profile.proto"],
    default_runtime = "@protobuf_archive//:protobuf_python",
    protoc = "@protobuf_archive//:protoc",
    srcs_version = "PY2AND3",
    deps = ["@protobuf_archive//:protobuf_python"],
)
