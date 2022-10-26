"""Provides a redirection point for platform specific implementations of Starlark utilities."""

load(
    "//tensorflow/core/profiler/builds/oss:build_config.bzl",
    _tf_profiler_alias = "tf_profiler_alias",
    _tf_profiler_pybind_cc_library_wrapper = "tf_profiler_pybind_cc_library_wrapper",
    _tf_profiler_xla_proto_header = "tf_profiler_xla_proto_header",
)
load(
    "//tensorflow/tsl/profiler/builds:build_config.bzl",
    _tf_profiler_copts = "tf_profiler_copts",
)

tf_profiler_alias = _tf_profiler_alias
tf_profiler_xla_proto_header = _tf_profiler_xla_proto_header
tf_profiler_pybind_cc_library_wrapper = _tf_profiler_pybind_cc_library_wrapper
tf_profiler_copts = _tf_profiler_copts

def if_profiler_oss(if_true, if_false = []):
    return select({
        "//tensorflow/core/profiler/builds:profiler_build_oss": if_true,
        "//conditions:default": if_false,
    })
