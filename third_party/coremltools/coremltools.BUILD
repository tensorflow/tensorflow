load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

exports_files(["LICENSE.txt"])

proto_library(
    name = "mlmodel_proto",
    srcs = glob(["mlmodel/format/*.proto"]),
)

cc_proto_library(
    name = "mlmodel_cc_proto",
    deps = [":mlmodel_proto"],
)
