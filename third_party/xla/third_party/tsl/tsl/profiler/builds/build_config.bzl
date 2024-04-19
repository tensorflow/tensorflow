"""Provides a redirection point for platform specific implementations of Starlark utilities."""

load("@local_xla//xla/tsl:tsl.bzl", "clean_dep")
load(
    "//tsl/profiler/builds/oss:build_config.bzl",
    _tf_profiler_alias = "tf_profiler_alias",
    _tf_profiler_pybind_cc_library_wrapper = "tf_profiler_pybind_cc_library_wrapper",
)

tf_profiler_pybind_cc_library_wrapper = _tf_profiler_pybind_cc_library_wrapper
tf_profiler_alias = _tf_profiler_alias

def tf_profiler_copts():
    return []

def if_profiler_oss(if_true, if_false = []):
    return select({
        clean_dep("//tsl/profiler/builds:profiler_build_oss"): if_true,
        "//conditions:default": if_false,
    })
