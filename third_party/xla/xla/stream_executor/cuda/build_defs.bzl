"""
This module contains custom build rules for CUDA assembly compiler tests.
"""

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

# Internally this loads a macro, but in OSS this is a function
# buildifier: disable=out-of-order-load
def register_extension_info(**_kwargs):
    pass

visibility(DEFAULT_LOAD_VISIBILITY + stream_executor_friends())

def _stage_in_bin_subdirectory_impl(ctx):
    if len(ctx.files.data) != 1:
        fail("Expected exactly one data dependency.")
    symlinks = {}
    symlinks["bin/" + ctx.label.name] = ctx.files.data[0]
    return [DefaultInfo(
        runfiles = ctx.runfiles(symlinks = symlinks),
    )]

# This rules takes a data dependency and makes it available under bin/<rule_name> in the runfiles
# directory. This is useful for some of our CUDA logic which expects to find binaries in a bin/
# subdirectory.
stage_in_bin_subdirectory = rule(
    implementation = _stage_in_bin_subdirectory_impl,
    attrs = {
        "data": attr.label_list(allow_files = True),
    },
)

def embeddable_cuda_library(**kwargs):
    """Wrapper around cuda_library that applies linkopts and deps for kernel registration.

    Args:
        **kwargs: Additional arguments to pass to cuda_library.
    """
    linkopts = kwargs.pop("linkopts", [])
    wrap_opts = [
        "-Wl,--wrap=__cudaRegisterFatBinary",
        "-Wl,--wrap=__cudaRegisterFunction",
    ]
    if type(linkopts) == "list":
        linkopts = list(linkopts)
        for opt in wrap_opts:
            if opt not in linkopts:
                linkopts.append(opt)
        kwargs["linkopts"] = linkopts
    else:
        kwargs["linkopts"] = wrap_opts + linkopts

    deps = kwargs.pop("deps", [])
    registry_dep = "//xla/stream_executor/cuda:cudart_kernel_registry"
    if type(deps) == "list":
        deps = list(deps)
        if registry_dep not in deps and ":cudart_kernel_registry" not in deps:
            deps.append(registry_dep)
        kwargs["deps"] = deps
    else:
        kwargs["deps"] = [registry_dep] + deps

    cuda_library(**kwargs)

register_extension_info(
    extension = embeddable_cuda_library,
    label_regex_for_dep = "{extension_name}",
)
