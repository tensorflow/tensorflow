load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

exports_files(["pprof/LICENSE"])

proto_library(
    name = "pprof_proto",
    srcs = ["proto/profile.proto"],
)

py_proto_library(
    name = "pprof_proto_py",
    deps = [":pprof_proto"],
)
