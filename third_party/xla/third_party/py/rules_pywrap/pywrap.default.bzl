# TODO(b/356020232): remove entire file and all usages after migration is done
"""
A proxy for pywrap.default.bzl. This file is maintained for backward compatibility and
sould be removed once all dependent projects have completed their migration.
"""

load(
    "@rules_ml_toolchain//py/rules_pywrap:pywrap.default.bzl",
    _pybind_extension = "pybind_extension",
    _pywrap_aware_cc_import = "pywrap_aware_cc_import",
    _pywrap_aware_filegroup = "pywrap_aware_filegroup",
    _pywrap_aware_genrule = "pywrap_aware_genrule",
    _pywrap_binaries = "pywrap_binaries",
    _pywrap_common_library = "pywrap_common_library",
    _pywrap_library = "pywrap_library",
    _stripped_cc_info = "stripped_cc_info",
    _use_pywrap_rules = "use_pywrap_rules",
)

# buildifier: disable=function-docstring-args
def pybind_extension(
        name,  # original
        deps,  # original
        srcs = [],  # original
        private_deps = [],  # original
        visibility = None,  # original
        win_def_file = None,  # original
        testonly = None,  # original
        compatible_with = None,  # original
        additional_exported_symbols = [],
        data = None,  # original
        # To patch top-level deps lists in sophisticated cases
        pywrap_ignored_deps_filter = ["@pybind11", "@pybind11//:pybind11"],
        local_defines = [],

        # Garbage parameters, exist only to maingain backward compatibility for
        # a while. Will be removed once migration is fully completed
        pytype_srcs = None,  # alias for data
        hdrs = [],  # merge into sources
        pytype_deps = None,  # ignore?
        ignore_link_in_framework = None,  # ignore
        dynamic_deps = [],  # ignore
        static_deps = [],  # ignore
        enable_stub_generation = None,  # ignore
        module_name = None,  # ignore
        link_in_framework = None,  # ignore
        additional_stubgen_deps = None,  # ignore
        **kwargs):
    _pybind_extension(
        name = name,
        deps = deps,
        srcs = srcs,
        private_deps = private_deps,
        visibility = visibility,
        win_def_file = win_def_file,
        testonly = testonly,
        compatible_with = compatible_with,
        additional_exported_symbols = additional_exported_symbols,
        data = data,
        pywrap_ignored_deps_filter = pywrap_ignored_deps_filter,
        local_defines = local_defines,
        pytype_srcs = pytype_srcs,
        hdrs = hdrs,
        pytype_deps = pytype_deps,
        ignore_link_in_framework = ignore_link_in_framework,
        dynamic_deps = dynamic_deps,
        static_deps = static_deps,
        enable_stub_generation = enable_stub_generation,
        module_name = module_name,
        link_in_framework = link_in_framework,
        additional_stubgen_deps = additional_stubgen_deps,
        **kwargs
    )

def use_pywrap_rules():
    return _use_pywrap_rules()

def pywrap_library(name, **kwargs):
    _pywrap_library(
        name = name,
        **kwargs
    )

def pywrap_common_library(name, **kwargs):
    _pywrap_common_library(
        name = name,
        **kwargs
    )

def stripped_cc_info(name, **kwargs):
    _stripped_cc_info(
        name = name,
        **kwargs
    )

def pywrap_aware_filegroup(name, **kwargs):
    _pywrap_aware_filegroup(
        name = name,
        **kwargs
    )

def pywrap_aware_genrule(name, **kwargs):
    _pywrap_aware_genrule(
        name = name,
        **kwargs
    )

def pywrap_aware_cc_import(name, **kwargs):
    _pywrap_aware_cc_import(
        name = name,
        **kwargs
    )

def pywrap_binaries(name, **kwargs):
    _pywrap_binaries(
        name = name,
        **kwargs
    )
