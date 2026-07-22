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

"""
This module contains custom build rules for CUDA assembly compiler tests.
"""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

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
