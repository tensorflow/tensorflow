# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Embed data files as TSL memfiles."""

load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/util:cc_embed_data.bzl", "cc_embed_data")

visibility("public")

def _gen_source_impl(ctx):
    args = ctx.actions.args()
    args.add("--generated_file=" + ctx.outputs.generated_file.path)
    args.add("--embed_dir=" + ctx.attr.embed_dir)
    args.add("--embed_name=" + ctx.attr.embed_name)
    args.add("--dirname=" + ctx.label.package)

    ctx.actions.run(
        outputs = [ctx.outputs.generated_file],
        arguments = [args],
        mnemonic = "MemfileEmbedDataGen",
        executable = ctx.executable._generator,
        progress_message = "Generating embedded memfile {}".format(ctx.label),
    )
    return [
        DefaultInfo(files = depset([ctx.outputs.generated_file])),
    ]

_gen_source = rule(
    implementation = _gen_source_impl,
    attrs = {
        "generated_file": attr.output(
            mandatory = True,
            doc = ".cc file to write the file contents to.",
        ),
        "embed_dir": attr.string(
            mandatory = True,
            doc = "Directory under embed:// to make the data files available at.",
        ),
        "embed_name": attr.string(
            mandatory = True,
            doc = "Name of the cc_embed_data target to include.",
        ),
        "_generator": attr.label(
            executable = True,
            cfg = "exec",
            default = Label("//xla/tsl/util:memfile_stub_generator"),
        ),
    },
)

def memfile_embed_data(
        name,
        srcs,
        embedded_dir_name = None,
        embedopts = None,
        flatten = None,
        testonly = None,
        compatible_with = None,
        **kwargs):
    """Embeds a list of input files in a binary.

    Example:
      memfile_embed_data(name = "mydata",
                         srcs = ["a.txt", "b/c.txt"])

    Any binary that links with mydata will be able to access a.txt and b/c.txt
    as embed://mydata/a.txt and embed://mydata/b/c.txt.

    Args:
      name: A unique name for this rule. (Name; required)
      srcs: The data files to be encapsulated. (List of labels; required)
      embedded_dir_name: If specified, the encapsulated data files will be
        accessible using paths like embed://embedded_dir_name/... instead of
        embed://name/... . (String; optional)
      embedopts: A list of options that will simply be passed through to
        filewrapper. (List of strings; optional; see cc_embed_data)
      flatten: If non-zero, the leading path components are removed from each srcs
        file. It overrides `strip`. (Boolean; optional; default is that of cc_embed_data)
      testonly: If true, this rule can only be included in test code.
      compatible_with: Forwarded to all underlying rules.
      **kwargs: Forwarded to the resulting cc_library.
    """
    embed_name = "%s_embed_internal" % name
    cc_embed_data(
        name = embed_name,
        srcs = srcs,
        outs = [
            embed_name + ".h",
            embed_name + ".cc",
        ],
        embedopts = embedopts,
        flatten = flatten,
        compatible_with = compatible_with,
        testonly = testonly,
        visibility = ["//visibility:private"],
        tags = ["manual"],
    )

    embedded_dir_name = name if not embedded_dir_name else embedded_dir_name
    cc_name = "%s_embed_internal_builtin.cc" % name
    _gen_source(
        name = "%s_embed_cc_src" % name,
        generated_file = cc_name,
        embed_dir = embedded_dir_name,
        embed_name = embed_name,
        testonly = testonly,
        compatible_with = compatible_with,
        tags = ["manual"],
    )

    cc_library(
        name = name,
        srcs = [cc_name],
        deps = [
            ":" + embed_name,
            "//xla/tsl/util:memfile_builtin",
        ],
        alwayslink = True,
        compatible_with = compatible_with,
        testonly = testonly,
        **kwargs
    )
