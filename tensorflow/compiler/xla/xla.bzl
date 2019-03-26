"""Wrapper around cc_proto_library used inside the XLA codebase."""

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "cc_proto_library",
)
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "if_static",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")

# xla_proto_library() is a convenience wrapper around cc_proto_library.
def xla_proto_library(name, srcs = [], deps = [], visibility = None, testonly = 0, **kwargs):
    if kwargs.get("use_grpc_plugin"):
        kwargs["use_grpc_namespace"] = True
    cc_proto_library(
        name = name,
        srcs = srcs,
        # Append well-known proto dep. As far as I know this is the only way
        # for xla_proto_library to access google.protobuf.{Any,Duration,...}.
        deps = deps + ["@protobuf_archive//:cc_wkt_protos"],
        cc_libs = if_static(
            ["@protobuf_archive//:protobuf"],
            otherwise = ["@protobuf_archive//:protobuf_headers"],
        ),
        protoc = "@protobuf_archive//:protoc",
        testonly = testonly,
        visibility = visibility,
        **kwargs
    )

def xla_py_proto_library(**kwargs):
    # Note: we don't currently define a proto library target for Python in OSS.
    _ignore = kwargs
    pass

def xla_py_grpc_library(**kwargs):
    # Note: we don't currently define any special targets for Python GRPC in OSS.
    _ignore = kwargs
    pass

ORC_JIT_MEMORY_MAPPER_TARGETS = []

# We link the GPU plugin into the XLA Python extension if CUDA is enabled.
def xla_python_default_plugins():
    return if_cuda_is_configured(["//tensorflow/compiler/xla/service:gpu_plugin"])
