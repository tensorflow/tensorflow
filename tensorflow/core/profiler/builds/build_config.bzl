"""Provides a redirection point for platform specific implementations of Starlark utilities."""

load(
    "//tensorflow/tsl/profiler/builds:build_config.bzl",
    _if_profiler_oss = "if_profiler_oss",
    _tf_profiler_alias = "tf_profiler_alias",
    _tf_profiler_copts = "tf_profiler_copts",
    _tf_profiler_pybind_cc_library_wrapper = "tf_profiler_pybind_cc_library_wrapper",
)

tf_profiler_alias = _tf_profiler_alias
tf_profiler_pybind_cc_library_wrapper = _tf_profiler_pybind_cc_library_wrapper
tf_profiler_copts = _tf_profiler_copts
if_profiler_oss = _if_profiler_oss
