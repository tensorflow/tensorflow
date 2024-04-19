load("@local_tsl//tsl/platform/default:build_config.bzl", "py_proto_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

exports_files(["pprof/LICENSE"])

py_proto_library(
    name = "pprof_proto_py",
    srcs = ["proto/profile.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY3",
    deps = ["@com_google_protobuf//:protobuf_python"],
)
