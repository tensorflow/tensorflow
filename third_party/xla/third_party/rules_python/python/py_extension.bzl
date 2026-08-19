"""Wrapper for py_extension in OSS using native.cc_binary."""

def py_extension(name, srcs, deps = [], copts = [], linkopts = [], **kwargs):
    # Remove features or other attributes unsupported by cc_binary
    kwargs.pop("features", None)

    native.cc_binary(
        name = name,
        srcs = srcs,
        copts = copts + ["-fexceptions", "-fPIC"],
        linkshared = True,
        linkopts = linkopts,
        deps = deps + [
            "@rules_python//python/cc:current_py_cc_headers",
        ],
        **kwargs
    )
