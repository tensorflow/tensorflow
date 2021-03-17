load("@rules_proto//proto:defs.bzl", "proto_library")
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

# Map of all well known protos.
# name => (include path, imports)
WELL_KNOWN_PROTO_MAP = {
    "any": ("google/protobuf/any.proto", []),
    "api": (
        "google/protobuf/api.proto",
        [
            "source_context",
            "type",
        ],
    ),
    "compiler_plugin": (
        "google/protobuf/compiler/plugin.proto",
        ["descriptor"],
    ),
    "descriptor": ("google/protobuf/descriptor.proto", []),
    "duration": ("google/protobuf/duration.proto", []),
    "empty": ("google/protobuf/empty.proto", []),
    "field_mask": ("google/protobuf/field_mask.proto", []),
    "source_context": ("google/protobuf/source_context.proto", []),
    "struct": ("google/protobuf/struct.proto", []),
    "timestamp": ("google/protobuf/timestamp.proto", []),
    "type": (
        "google/protobuf/type.proto",
        [
            "any",
            "source_context",
        ],
    ),
    "wrappers": ("google/protobuf/wrappers.proto", []),
}

RELATIVE_WELL_KNOWN_PROTOS = [proto[1][0] for proto in WELL_KNOWN_PROTO_MAP.items()]

genrule(
    name = "link_proto_files",
    outs = RELATIVE_WELL_KNOWN_PROTOS,
    cmd = """
      for i in $(OUTS); do
        f=$${i#$(@D)/}
        mkdir -p $(@D)/$${f%/*}
        ln -sf $(PROTOBUF_INCLUDE_PATH)/$$f $(@D)/$$f
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
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

[proto_library(
    name = proto[0] + "_proto",
    srcs = [proto[1][0]],
    visibility = ["//visibility:public"],
    deps = [dep + "_proto" for dep in proto[1][1]],
) for proto in WELL_KNOWN_PROTO_MAP.items()]
