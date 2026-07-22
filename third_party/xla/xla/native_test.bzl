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

"""Native test rule for LIT tests."""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def _shared_impl(ctx):
    out = ctx.attr.out
    if not out:
        out = ctx.attr.name
    output = ctx.actions.declare_file(out)
    ctx.actions.symlink(
        target_file = ctx.executable.src,
        output = output,
        is_executable = True,
    )

    runfiles = ctx.runfiles(files = ctx.files.data)

    # For Bazel 4.x support. Drop when Bazel 4.x is no longer supported
    to_merge = ([d[DefaultInfo].default_runfiles for d in ctx.attr.data] +
                [ctx.attr.src[DefaultInfo].default_runfiles])
    if hasattr(runfiles, "merge_all"):
        runfiles = runfiles.merge_all(to_merge)
    else:
        for m in to_merge:
            runfiles = runfiles.merge(m)
    return DefaultInfo(
        executable = output,
        files = depset([output]),
        runfiles = runfiles,
    )

def _native_test_impl(ctx):
    default_info = _shared_impl(ctx)
    return [default_info, testing.TestEnvironment(ctx.attr.env)]

_TEST_ATTRS = {
    "src": attr.label(
        executable = True,
        allow_files = True,
        mandatory = True,
        cfg = "target",
    ),
    "data": attr.label_list(allow_files = True),
    # "out" is attr.string instead of attr.output, so that it is select()'able.
    "out": attr.string(),
    "env": attr.string_dict(
        doc = "Mirrors the common env attribute that otherwise is" +
              " only available on native rules. See" +
              " https://docs.bazel.build/versions/main/be/common-definitions.html#test.env",
    ),
}

native_test = rule(
    implementation = _native_test_impl,
    attrs = _TEST_ATTRS,
    test = True,
    doc = "Rule to create a symlink to a test executable for LIT tests.",
)
