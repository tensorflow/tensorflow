"""
A proxy for pywrap.impl.bzl. This file is maintained for backward compatibility and
sould be removed once all dependent projects have completed their migration.
"""

load(
    "@rules_ml_toolchain//py/rules_pywrap:pywrap.impl.bzl",
    _collected_pywrap_infos = "collected_pywrap_infos",
    _generated_common_win_def_file = "generated_common_win_def_file",
    _pybind_extension = "pybind_extension",
    _pywrap_binaries = "pywrap_binaries",
    _pywrap_common_library = "pywrap_common_library",
    _pywrap_library = "pywrap_library",
    _stripped_cc_info = "stripped_cc_info",
)

def pywrap_library(
        name,
        deps,
        starlark_only_deps = [],
        # Makes no sense for wrapped PyInit_ case, and should not be used
        pywrap_lib_filter = None,
        pywrap_lib_exclusion_filter = None,
        common_lib_filters = {},
        common_lib_versions = {},
        common_lib_version_scripts = {},
        common_lib_def_files_or_filters = {},
        common_lib_linkopts = {},
        enable_common_lib_starlark_only_filter = True,
        pywrap_count = None,
        starlark_only_pywrap_count = 0,
        extra_deps = ["@pybind11//:pybind11"],
        visibility = None,
        testonly = None,
        compatible_with = None):
    _pywrap_library(
        name = name,
        deps = deps,
        starlark_only_deps = starlark_only_deps,
        pywrap_lib_filter = pywrap_lib_filter,
        pywrap_lib_exclusion_filter = pywrap_lib_exclusion_filter,
        common_lib_filters = common_lib_filters,
        common_lib_versions = common_lib_versions,
        common_lib_version_scripts = common_lib_version_scripts,
        common_lib_def_files_or_filters = common_lib_def_files_or_filters,
        common_lib_linkopts = common_lib_linkopts,
        enable_common_lib_starlark_only_filter = enable_common_lib_starlark_only_filter,
        pywrap_count = pywrap_count,
        starlark_only_pywrap_count = starlark_only_pywrap_count,
        extra_deps = extra_deps,
        visibility = visibility,
        testonly = testonly,
        compatible_with = compatible_with,
    )

def pywrap_common_library(name, dep, filter_name = None, **kwargs):
    _pywrap_common_library(
        name = name,
        dep = dep,
        filter_name = filter_name,
        **kwargs
    )

def pywrap_binaries(name, dep, **kwargs):
    _pywrap_binaries(
        name = name,
        dep = dep,
        **kwargs
    )

def python_extension(
        name,
        deps,
        srcs = [],
        common_lib_packages = [],
        visibility = None,
        win_def_file = None,
        testonly = None,
        compatible_with = None,
        additional_exported_symbols = [],
        default_deps = [],
        linkopts = [],
        starlark_only = False,
        local_defines = [],
        wrap_py_init = None,
        **kwargs):
    _pybind_extension(
        name = name,
        deps = deps,
        srcs = srcs,
        common_lib_packages = common_lib_packages,
        visibility = visibility,
        win_def_file = win_def_file,
        testonly = testonly,
        compatible_with = compatible_with,
        additional_exported_symbols = additional_exported_symbols,
        default_deps = default_deps,
        linkopts = linkopts,
        starlark_only = starlark_only,
        local_defines = local_defines,
        wrap_py_init = wrap_py_init,
        **kwargs
    )

# For backward compatibility with the old name
def pybind_extension(name, default_deps = None, **kwargs):
    _pybind_extension(
        name = name,
        default_deps = default_deps,
        **kwargs
    )

def collected_pywrap_infos(name, **kwargs):
    _collected_pywrap_infos(
        name = name,
        **kwargs
    )

def stripped_cc_info(name, **kwargs):
    _stripped_cc_info(
        name = name,
        **kwargs
    )

def generated_common_win_def_file(name, **kwargs):
    _generated_common_win_def_file(
        name = name,
        **kwargs
    )
