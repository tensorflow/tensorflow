load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

exports_files(["LICENSE.txt"])

cc_proto_library(
    name = "mlmodel_cc_proto",
    srcs = glob(["mlmodel/format/*.proto"]),
    include = "mlmodel/format",
    default_runtime = "@com_google_protobuf//:protobuf_lite",
    protoc = "@com_google_protobuf//:protoc",
)
