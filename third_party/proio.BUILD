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
        "@lz4_archive//:lz4",
        ":proio.pb_cc",
        ],
    includes = [
        "proto",
        ],
)

load(
    "@org_tensorflow//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

tf_proto_library(
    name = "proio.pb",
#    srcs = glob(["*.proto"]),
    srcs = ["proto/proio.proto"] + glob(["model/*.proto"]),
)
