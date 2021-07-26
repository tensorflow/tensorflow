# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""BUILD extensions for MLIR linalg generation."""

def genlinalg(name, linalggen, src, linalg_outs):
    """genlinalg() generates code from a tc spec file.

    Args:
      name: The name of the build rule for use in dependencies.
      linalggen: The binary used to produce the output.
      src: The tc spec file.
      linalg_outs: A list of tuples (opts, out), where each opts is a string of
        options passed to linalggen, and the out is the corresponding output file
        produced.
    """

    for (opts, out) in linalg_outs:
        # All arguments to generate the output except output destination.
        base_args = [
            "$(location %s)" % linalggen,
            "%s" % opts,
            "$(location %s)" % src,
        ]
        rule_suffix = "_".join(opts.replace("-", "_").replace("=", "_").split(" "))

        # Rule to generate code using generated shell script.
        native.genrule(
            name = "%s_%s_genrule" % (name, rule_suffix),
            srcs = [src],
            outs = [out],
            tools = [linalggen],
            cmd = (" ".join(base_args)),
        )

    hdrs = [f for (opts, f) in linalg_outs]
    native.cc_library(
        name = name,
        hdrs = hdrs,
        textual_hdrs = hdrs,
    )
