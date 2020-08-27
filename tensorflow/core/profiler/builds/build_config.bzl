"""Provides a redirection point for platform specific implementations of Starlark utilities."""

load(
    "//tensorflow/core/profiler/builds/oss:build_config.bzl",
    _tf_profiler_alias = "tf_profiler_alias",
)

tf_profiler_alias = _tf_profiler_alias

def if_profiler_oss(if_true, if_false = []):
    return select({
        "//tensorflow/core/profiler/builds:profiler_build_oss": if_true,
        "//conditions:default": if_false,
    })
