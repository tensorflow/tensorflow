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

"""Starlark macros for oneDNN Graph API.

Contains library and test rules that builds with empty srcs, hdrs, and deps if not build with Graph
API or oneDNN. These rules have to be outside of mkl/build_defs.bzl, otherwise we would have cyclic
dependency (xla.bzl depends on tsl which depends on mkl/build_defs.bzl).

TODO(penporn): Rename this file to build_rules.bzl since it's not just about graph API anymore.
"""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load("//xla/tsl/mkl:build_defs.bzl", "if_graph_api", "if_onednn", "if_onednn_async")

visibility(DEFAULT_LOAD_VISIBILITY)

def onednn_graph_cc_library(srcs = [], hdrs = [], deps = [], **kwargs):
    """cc_library rule that has empty src, hdrs and deps if not building with Graph API."""
    cc_library(
        srcs = if_graph_api(srcs),
        hdrs = if_graph_api(hdrs),
        deps = if_graph_api(deps),
        **kwargs
    )

def onednn_graph_cc_test(
        srcs = [],
        deps = [],
        **kwargs):
    """xla_cc_test rule that has empty src and deps if not building with Graph API."""
    xla_cc_test(
        srcs = if_graph_api(srcs),
        deps = if_graph_api(if_true = deps, if_false = ["//xla/tsl/platform:test_main"]),
        # If not building with Graph API, we don't have any tests linked.
        fail_if_no_test_linked = False,
        # If not building with Graph API, we don't have any tests defined either.
        fail_if_no_test_selected = False,
        **kwargs
    )

def onednn_cc_library(srcs = [], hdrs = [], deps = [], **kwargs):
    """cc_library rule with empty src/hdrs/deps if not building with oneDNN."""
    cc_library(
        srcs = if_onednn(srcs),
        hdrs = if_onednn(hdrs),
        deps = if_onednn(deps),
        # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
        **kwargs
    )

def onednn_cc_test(
        srcs = [],
        deps = [],
        **kwargs):
    """xla_cc_test rule with empty src and deps if not building with oneDNN."""
    xla_cc_test(
        # CC_TEST_OK=This rule is used in XLA.
        srcs = if_onednn_async(srcs),
        deps = if_onednn_async(if_true = deps, if_false = ["//xla/tsl/platform:test_main"]),
        # If not building with Graph API, we don't have any tests linked.
        fail_if_no_test_linked = False,
        # If not building with Graph API, we don't have any tests defined either.
        fail_if_no_test_selected = False,
        **kwargs
    )
