"""Wrapper around cc_proto_library used inside the XLA codebase."""

load("//tensorflow/core:platform/default/build_config.bzl",
     "cc_proto_library")
load("//tensorflow/core:platform/default/build_config_root.bzl",
     "if_static")

# xla_proto_library() is a convenience wrapper around cc_proto_library.
def xla_proto_library(name, srcs=[], deps=[], visibility=None, testonly=0):
  cc_proto_library(name=name,
                   srcs=srcs,
                   deps=deps,
                   cc_libs = if_static(
                       ["@protobuf_archive//:protobuf"],
                       otherwise=["@protobuf_archive//:protobuf_headers"],
                   ),
                   protoc="@protobuf_archive//:protoc",
                   testonly=testonly,
                   visibility=visibility,)

# Flags required for modules that export symbols that are to be called by the
# XLA CustomCall operator. CustomCall must be able to find symbols with dlsym(),
# which on Linux requires we link with --export-dynamic.
export_dynamic_linkopts = select({
    "//tensorflow:darwin": [],
    "//conditions:default": ["-Wl,--export-dynamic"],
})
