# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules and macros for MLIR"""

def if_cuda_available(if_true, if_false = []):
    return select({
        # CUDA is not yet supported.
        "//conditions:default": if_false,
    })

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
