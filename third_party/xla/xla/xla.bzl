"""Wrapper around proto libraries used inside the XLA codebase."""

load(
    "//xla/tsl:tsl.bzl",
    "tsl_copts",
)

def xla_py_proto_library(**_kwargs):
    # Note: we don't currently define a proto library target for Python in OSS.
    pass

def xla_py_test_deps():
    return []

def xla_cc_binary(deps = [], copts = tsl_copts(), **kwargs):
    native.cc_binary(deps = deps, copts = copts, **kwargs)

def xla_cc_test(name, **kwargs):
    native.cc_test(
        name = name,
        **kwargs
    )

def xla_nvml_deps():
    return ["@local_config_cuda//cuda:nvml_headers"]

def xla_cub_deps():
    return ["@local_config_cuda//cuda:cub_headers"]

def xla_internal(targets, otherwise = []):
    _ = targets  # buildifier: disable=unused-variable
    return otherwise

def tests_build_defs_bzl_deps():
    return []
