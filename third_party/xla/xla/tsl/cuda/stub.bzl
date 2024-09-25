"""Macros to generate CUDA library stubs from a list of symbols."""

def cuda_stub(name, srcs):
    """Generates a CUDA stub from a list of symbols.

    Generates two files:
    * library.inc, which contains a list of symbols suitable for inclusion by
        C++, and
    * library.tramp.S, which contains assembly-language trampolines for each
      symbol.
    """
    native.genrule(
        name = "{}_stub_gen".format(name),
        srcs = srcs,
        tools = ["//third_party/implib_so:make_stub"],
        outs = [
            "{}.inc".format(name),
            "{}.tramp.S".format(name),
        ],
        tags = ["gpu"],
        cmd = select({
            "@local_xla//xla/tsl:linux_aarch64": "$(location //third_party/implib_so:make_stub) $< --outdir $(RULEDIR) --target aarch64",
            "@local_xla//xla/tsl:linux_x86_64": "$(location //third_party/implib_so:make_stub) $< --outdir $(RULEDIR) --target x86_64",
            "@local_xla//xla/tsl:linux_ppc64le": "$(location //third_party/implib_so:make_stub) $< --outdir $(RULEDIR) --target powerpc64le",
            "//conditions:default": "NOT_IMPLEMENTED_FOR_THIS_PLATFORM_OR_ARCHITECTURE",
        }),
    )
