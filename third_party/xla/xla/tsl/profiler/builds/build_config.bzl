"""Provides a redirection point for platform specific implementations of Starlark utilities."""

load(
    "//xla/tsl:package_groups.bzl",
    "DEFAULT_LOAD_VISIBILITY",
    "LEGACY_TSL_PROFILER_BUILDS_BUILD_CONFIG_USERS",
)
load("//xla/tsl:tsl.bzl", "clean_dep")
load(
    "//xla/tsl/profiler/builds/oss:build_config.bzl",
    _tf_profiler_alias = "tf_profiler_alias",
    _tf_profiler_pybind_cc_library_wrapper = "tf_profiler_pybind_cc_library_wrapper",
)

visibility(DEFAULT_LOAD_VISIBILITY + LEGACY_TSL_PROFILER_BUILDS_BUILD_CONFIG_USERS)

tf_profiler_pybind_cc_library_wrapper = _tf_profiler_pybind_cc_library_wrapper
tf_profiler_alias = _tf_profiler_alias

def tf_profiler_copts():
    return []

def if_profiler_oss(if_true, if_false = []):
    return select({
        clean_dep("//xla/tsl/profiler/builds:profiler_build_oss"): if_true,
        "//conditions:default": if_false,
    })
