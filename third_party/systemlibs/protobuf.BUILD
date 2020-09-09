load(
    "@com_google_protobuf//:protobuf.bzl",
    "cc_proto_library",
    "proto_gen",
    "py_proto_library",
)

licenses(["notice"])

filegroup(
    name = "LICENSE",
    visibility = ["//visibility:public"],
)

HEADERS = [
    "google/protobuf/any.pb.h",
    "google/protobuf/any.proto",
    "google/protobuf/api.pb.h",
    "google/protobuf/api.proto",
    "google/protobuf/arena.h",
    "google/protobuf/compiler/importer.h",
    "google/protobuf/compiler/plugin.h",
    "google/protobuf/compiler/plugin.pb.h",
    "google/protobuf/compiler/plugin.proto",
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
    "google/protobuf/repeated_field.h",
    "google/protobuf/source_context.pb.h",
    "google/protobuf/source_context.proto",
    "google/protobuf/struct.pb.h",
    "google/protobuf/struct.proto",
    "google/protobuf/text_format.h",
    "google/protobuf/timestamp.pb.h",
    "google/protobuf/timestamp.proto",
    "google/protobuf/type.pb.h",
    "google/protobuf/type.proto",
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
    protoc = "@com_google_protobuf//:protoc",
    visibility = ["//visibility:public"],
)

py_library(
    name = "protobuf_python",
    data = [":link_headers"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

proto_library(
    name = "any_proto",
    srcs = ["google/protobuf/any.proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "api_proto",
    srcs = ["google/protobuf/api.proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "compiler_plugin_proto",
    srcs = ["google/protobuf/compiler/plugin.proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "descriptor_proto",
    srcs = ["google/protobuf/descriptor.proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "duration_proto",
    srcs = ["google/protobuf/duration.proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "empty_proto",
    srcs = ["google/protobuf/empty.proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "field_mask_proto",
    srcs = ["google/protobuf/field_mask.proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "source_context_proto",
    srcs = ["google/protobuf/source_context.proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "struct_proto",
    srcs = ["google/protobuf/struct.proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "timestamp_proto",
    srcs = ["google/protobuf/timestamp.proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "type_proto",
    srcs = ["google/protobuf/type.proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "wrappers_proto",
    srcs = ["google/protobuf/wrappers.proto"],
    visibility = ["//visibility:public"],
)
