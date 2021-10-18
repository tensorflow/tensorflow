"""Starlark macros for building pasta."""

def copy_srcs(srcs):
    """Copies srcs from 'pasta' to parent directory."""
    for src in srcs:
        native.genrule(
            name = src.replace(".", "_"),
            srcs = ["pasta/" + src],
            outs = [src],
            cmd = "mkdir -p $$(dirname $@); cp $< $@",
        )
    return srcs
