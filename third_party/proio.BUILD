package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "proio",
    srcs = [
        "cpp-proio/src/event.cc",
        "cpp-proio/src/reader.cc",
        "cpp-proio/src/writer.cc",
        ],
    hdrs = [
        "cpp-proio/src/event.h",
        "cpp-proio/src/reader.h",
        "cpp-proio/src/writer.h",
        ],
    deps = [
        "@protobuf_archive//:protobuf",
        "@protobuf_archive//:protobuf_lite",
        "@lz4_archive//:lz4",
        ":proio.pb",
        ],
    includes = [
        "proto",
        "cpp-proio/src",
        ],
)

load(
    "@protobuf_archive//:protobuf.bzl",
    "cc_proto_library",
)

cc_proto_library(
    name = "proio.pb",
    srcs = [
        "proto/proio.proto",
        ],
    protoc = "@protobuf_archive//:protoc",
    default_runtime = "@protobuf_archive//:protobuf",
)
