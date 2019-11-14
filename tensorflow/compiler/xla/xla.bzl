"""Wrapper around cc_proto_library used inside the XLA codebase."""

load(
    "//tensorflow/core/platform:default/build_config.bzl",
    "tf_proto_library_cc",
)

# xla_proto_library() is a convenience wrapper around cc_proto_library.
def xla_proto_library(name, srcs = [], deps = [], visibility = None, testonly = 0, **kwargs):
    if kwargs.pop("use_grpc_plugin", None):
        kwargs["use_grpc_namespace"] = True
        kwargs["cc_grpc_version"] = 1
    tf_proto_library_cc(
        name = name,
        srcs = srcs,
        protodeps = deps,
        cc_api_version = 2,
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

def xla_py_test_deps():
    return []
