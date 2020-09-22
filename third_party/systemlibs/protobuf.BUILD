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

PROTO_FILES = [
    "google/protobuf/any.proto",
    "google/protobuf/api.proto",
    "google/protobuf/compiler/plugin.proto",
    "google/protobuf/descriptor.proto",
    "google/protobuf/duration.proto",
    "google/protobuf/empty.proto",
    "google/protobuf/field_mask.proto",
    "google/protobuf/source_context.proto",
    "google/protobuf/struct.proto",
    "google/protobuf/timestamp.proto",
    "google/protobuf/type.proto",
    "google/protobuf/wrappers.proto",
]

genrule(
    name = "link_proto_files",
    outs = PROTO_FILES,
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
    linkopts = ["-lprotobuf"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "protobuf_headers",
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
