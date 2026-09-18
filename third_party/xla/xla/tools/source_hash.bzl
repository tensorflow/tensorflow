# Copyright 2026 The OpenXLA Authors.
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
"""Library to compute the hash of a target.

Computes a deterministic hash of a binary for version sealing.
"""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//xla:xla.default.bzl", "xla_cc_binary")

visibility("public")

def _source_hash_impl(ctx):
    args = ctx.actions.args()
    args.add(ctx.file.target)
    args.add(ctx.outputs.out)

    ctx.actions.run(
        inputs = [ctx.file.target],
        outputs = [ctx.outputs.out],
        executable = ctx.executable._hasher,
        arguments = [args],
        mnemonic = "ComputeSourceHash",
    )

    return [
        DefaultInfo(files = depset([ctx.outputs.out])),
    ]

def _verify_alwayslink_aspect_impl(target, ctx):
    if ctx.rule.kind == "alias":
        return []

    if CcInfo not in target:
        fail("Target %s passed to source_hash must provide CcInfo" % target.label)

    cc_info = target[CcInfo]
    linking_context = cc_info.linking_context

    has_direct_libs = False

    for linker_input in linking_context.linker_inputs.to_list():
        if linker_input.owner == target.label:
            for lib in linker_input.libraries:
                has_direct_libs = True
                if (lib.static_library or lib.pic_static_library) and not lib.alwayslink:
                    fail("Target %s passed to source_hash must be alwayslink" % target.label)

    if not has_direct_libs:
        fail("Target %s passed to source_hash does not produce any direct libraries to link" % target.label)

    return []

_verify_alwayslink_aspect = aspect(
    implementation = _verify_alwayslink_aspect_impl,
    attr_aspects = ["actual"],
)

_source_hash_rule = rule(
    implementation = _source_hash_impl,
    attrs = {
        "target": attr.label(
            mandatory = True,
            executable = True,
            allow_single_file = True,
            providers = [CcInfo],
            cfg = "exec",
            doc = "Executable target to hash.",
        ),
        "original_target": attr.label(
            mandatory = True,
            providers = [CcInfo],
            aspects = [_verify_alwayslink_aspect],
            cfg = "exec",
        ),
        "_hasher": attr.label(
            default = "//xla/tools:source_hash_hasher",
            executable = True,
            cfg = "exec",
        ),
        "out": attr.output(
            mandatory = True,
            doc = "Text file to write the hex hash to.",
        ),
    },
)

def source_hash(name, target, out):
    """Generates a text file containing a hash of the executable target.

    Args:
      name: string, Name of the rule.
      target: label, cc_library to hash. Must be alwayslink.
      out: string, Name of the file to write the hash to.
    """
    cc_name = name + "_source_hash_internal.cc"
    native.genrule(
        name = "_%s_write_source_hash_internal" % cc_name,
        outs = [cc_name],
        cmd = "echo 'int main() { return 0; }' > \"$@\"",
    )

    binary_name = "_%s_source_hash_internal" % name
    xla_cc_binary(
        name = binary_name,
        srcs = [cc_name],
        deps = [target],
        stamp = 0,
    )

    _source_hash_rule(
        name = name,
        target = binary_name,
        original_target = target,
        out = out,
    )
