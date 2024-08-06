# TODO(b/356020232): remove entire file and all usages after migration is done
load("//third_party/py/rules_pywrap:pywrap.bzl", "pybind_extension")

def pywrap_pybind_extension(
        name, # original
        deps, # original
        srcs = [], # original
        private_deps = [], # original
        visibility = None, # original
        win_def_file = None, # original
        testonly = None, # original
        compatible_with = None, # original
        outer_module_name = "", # deprecate
        additional_exported_symbols = [],
        data = None, # original
        # Garbage parameters, exist only to maingain backward compatibility for
        # a while. Will be removed once migration is fully completed

        # To patch top-level deps lists in sophisticated cases
        pywrap_ignored_deps_filter = ["@pybind11", "@pybind11//:pybind11"],
        pywrap_private_deps_filter = [
            "@pybind11_abseil//pybind11_abseil:absl_casters",
            "@pybind11_abseil//pybind11_abseil:import_status_module",
            "@pybind11_abseil//pybind11_abseil:status_casters",
            "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        ],

        pytype_srcs = None, # alias for data
        hdrs = [], # merge into sources
        pytype_deps = None, # ignore?
        ignore_link_in_framework = None, # ignore
        dynamic_deps = [], # ignore
        static_deps = [], # ignore
        enable_stub_generation = None, # ignore
        module_name = None, # ignore
        link_in_framework = None, # ignore
        additional_stubgen_deps = None, # ignore
        **kwargs):

    _ignore = [
        ignore_link_in_framework,
        dynamic_deps,
        static_deps,
        enable_stub_generation,
        module_name,
        link_in_framework,
        additional_stubgen_deps,
        pytype_deps,
    ]

    private_deps_filter_dict = {k: None for k in pywrap_private_deps_filter}
    ignored_deps_filter_dict = {k: None for k in pywrap_ignored_deps_filter}

    actual_srcs = srcs + hdrs

    actual_data = data
    if pytype_srcs:
        data = pytype_srcs

    actual_deps = []
    actual_private_deps = []
    actual_default_deps = ["@pybind11//:pybind11"]

    if type(deps) == list:
        for dep in deps:
            if dep in ignored_deps_filter_dict:
                continue
            if dep in private_deps_filter_dict:
                actual_private_deps.append(dep)
                continue
            actual_deps.append(dep)
    else:
        actual_deps = deps
        actual_default_deps = []

    pybind_extension(
        name = name,
        deps = actual_deps,
        srcs = actual_srcs,
        private_deps = actual_private_deps,
        visibility = visibility,
        win_def_file = win_def_file,
        testonly = testonly,
        compatible_with = compatible_with,
        outer_module_name = outer_module_name,
        additional_exported_symbols = additional_exported_symbols,
        data = actual_data,
        default_deps = actual_default_deps,
        **kwargs,
    )

def _pywrap_compat_init_impl(repository_ctx):
    repository_ctx.file("BUILD", "")
    use_pywrap_rules = bool(
        repository_ctx.os.environ.get("USE_PYWRAP_RULES", False))

    if use_pywrap_rules:
        print("!!!Using pywrap rules instead of directly creating .so objects!!!")

    repository_ctx.file(
        "pywrap_compat.bzl",
"""
def use_pywrap_rules():
    return {}
""".format(use_pywrap_rules),
    )

pywrap_compat_init = repository_rule(
    implementation = _pywrap_compat_init_impl,
    environ = ["USE_PYWRAP_RULES"],
)
