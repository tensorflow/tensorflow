# Copyright 2026 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transitive rules for collecting headers and parameters from a set of dependencies."""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "//xla/tsl:package_groups.bzl",
    "DEFAULT_LOAD_VISIBILITY",
    "LEGACY_TSL_TSL_USERS",
)

visibility(DEFAULT_LOAD_VISIBILITY + LEGACY_TSL_TSL_USERS)

def _transitive_parameters_library_impl(ctx):
    defines = depset(
        transitive = [dep[CcInfo].compilation_context.defines for dep in ctx.attr.original_deps],
    )
    system_includes = depset(
        transitive = [dep[CcInfo].compilation_context.system_includes for dep in ctx.attr.original_deps],
    )
    includes = depset(
        transitive = [dep[CcInfo].compilation_context.includes for dep in ctx.attr.original_deps],
    )
    quote_includes = depset(
        transitive = [dep[CcInfo].compilation_context.quote_includes for dep in ctx.attr.original_deps],
    )
    framework_includes = depset(
        transitive = [dep[CcInfo].compilation_context.framework_includes for dep in ctx.attr.original_deps],
    )
    return CcInfo(
        compilation_context = cc_common.create_compilation_context(
            defines = depset(direct = defines.to_list()),
            system_includes = depset(direct = system_includes.to_list()),
            includes = depset(direct = includes.to_list()),
            quote_includes = depset(direct = quote_includes.to_list()),
            framework_includes = depset(direct = framework_includes.to_list()),
        ),
    )

# Bazel rule for collecting the transitive parameters from a set of dependencies into a library.
# Propagates defines and includes.
transitive_parameters_library = rule(
    attrs = {
        "original_deps": attr.label_list(
            allow_empty = True,
            allow_files = True,
            providers = [CcInfo],
        ),
    },
    implementation = _transitive_parameters_library_impl,
)

def _get_transitive_headers(hdrs, deps):
    """Obtain the header files for a target and its transitive dependencies.

      Args:
        hdrs: a list of header files
        deps: a list of targets that are direct dependencies

      Returns:
        a collection of the transitive headers
      """
    return depset(
        hdrs,
        transitive = [dep[CcInfo].compilation_context.headers for dep in deps],
    )

def _transitive_hdrs_impl(ctx):
    outputs = _get_transitive_headers([], ctx.attr.deps)
    return DefaultInfo(files = outputs)

# Bazel rule for collecting the header files that a target depends on.
transitive_hdrs = rule(
    attrs = {
        "deps": attr.label_list(
            allow_files = True,
            providers = [CcInfo],
        ),
    },
    implementation = _transitive_hdrs_impl,
)
