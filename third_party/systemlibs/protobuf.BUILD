load(
    "@protobuf_archive//:protobuf.bzl",
    "proto_gen",
    "py_proto_library",
    "cc_proto_library",
)

licenses(["notice"])

filegroup(
    name = "LICENSE",
    visibility = ["//visibility:public"],
)

HEADERS = [
    "google/protobuf/any.pb.h",
    "google/protobuf/any.proto",
    "google/protobuf/arena.h",
    "google/protobuf/compiler/importer.h",
    "google/protobuf/descriptor.h",
    "google/protobuf/descriptor.pb.h",
    "google/protobuf/descriptor.proto",
    "google/protobuf/duration.pb.h",
    "google/protobuf/duration.proto",
    "google/protobuf/dynamic_message.h",
    "google/protobuf/empty.pb.h",
    "google/protobuf/empty.proto",
    "google/protobuf/field_mask.pb.h",
    "google/protobuf/field_mask.proto",
    "google/protobuf/io/coded_stream.h",
    "google/protobuf/io/zero_copy_stream.h",
    "google/protobuf/io/zero_copy_stream_impl_lite.h",
    "google/protobuf/map.h",
    "google/protobuf/port_def.inc",
    "google/protobuf/repeated_field.h",
    "google/protobuf/text_format.h",
    "google/protobuf/timestamp.pb.h",
    "google/protobuf/timestamp.proto",
    "google/protobuf/util/json_util.h",
    "google/protobuf/util/type_resolver_util.h",
    "google/protobuf/wrappers.pb.h",
    "google/protobuf/wrappers.proto",
]

genrule(
    name = "link_headers",
    outs = HEADERS,
    cmd = """
      for i in $(OUTS); do
        f=$${i#$(@D)/}
        mkdir -p $(@D)/$${f%/*}
        ln -sf $(INCLUDEDIR)/$$f $(@D)/$$f
      done
    """,
)

cc_library(
    name = "protobuf",
    hdrs = HEADERS,
    linkopts = ["-lprotobuf"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "protobuf_headers",
    hdrs = HEADERS,
    linkopts = ["-lprotobuf"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "protoc_lib",
    linkopts = ["-lprotoc"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "protoc",
    outs = ["protoc.bin"],
    cmd = "ln -s $$(which protoc) $@",
    executable = 1,
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "cc_wkt_protos",
    hdrs = HEADERS,
    internal_bootstrap_hack = 1,
    protoc = ":protoc",
    visibility = ["//visibility:public"],
)

proto_gen(
    name = "protobuf_python_genproto",
    includes = ["."],
    protoc = "@protobuf_archive//:protoc",
    visibility = ["//visibility:public"],
)

py_library(
    name = "protobuf_python",
    data = [":link_headers"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)
