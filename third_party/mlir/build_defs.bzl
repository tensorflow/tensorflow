"""Rules and macros for MLIR"""

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")

if_cuda_available = if_cuda_is_configured

def _cc_headers_only_impl(ctx):
    return CcInfo(compilation_context = ctx.attr.src[CcInfo].compilation_context)

cc_headers_only = rule(
    implementation = _cc_headers_only_impl,
    attrs = {
        "src": attr.label(
            mandatory = True,
            providers = [CcInfo],
        ),
    },
    doc = "Provides the headers from 'src' without linking anything.",
    provides = [CcInfo],
)
