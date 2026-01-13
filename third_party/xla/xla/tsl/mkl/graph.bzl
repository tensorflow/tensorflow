"""Starlark macros for oneDNN Graph API.

Contains library and test rules that builds with empty srcs, hdrs, and deps if not build with Graph
API or oneDNN. These rules have to be outside of mkl/build_defs.bzl, otherwise we would have cyclic
dependency (xla.bzl depends on tsl which depends on mkl/build_defs.bzl).

TODO(penporn): Rename this file to build_rules.bzl since it's not just about graph API anymore.
"""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load("//xla/tsl/mkl:build_defs.bzl", "if_graph_api", "if_onednn")

# Internally this loads a macro, but in OSS this is a function
# buildifier: disable=out-of-order-load
def register_extension_info(**_kwargs):
    pass

visibility(DEFAULT_LOAD_VISIBILITY)

def onednn_graph_cc_library(srcs = [], hdrs = [], deps = [], **kwargs):
    """cc_library rule that has empty src, hdrs and deps if not building with Graph API."""
    cc_library(
        srcs = if_graph_api(srcs),
        hdrs = if_graph_api(hdrs),
        deps = if_graph_api(deps),
        **kwargs
    )

register_extension_info(
    extension = onednn_graph_cc_library,
    label_regex_for_dep = "{extension_name}",
)

def onednn_graph_cc_test(
        srcs = [],
        deps = [],
        **kwargs):
    """xla_cc_test rule that has empty src and deps if not building with Graph API."""
    xla_cc_test(
        srcs = if_graph_api(srcs),
        deps = if_graph_api(if_true = deps, if_false = ["@com_google_googletest//:gtest_main"]),
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

register_extension_info(
    extension = onednn_cc_library,
    label_regex_for_dep = "{extension_name}",
)

def onednn_cc_test(
        srcs = [],
        deps = [],
        **kwargs):
    """xla_cc_test rule with empty src and deps if not building with Graph API."""
    xla_cc_test(
        # CC_TEST_OK=This rule is used in XLA.
        srcs = if_onednn(srcs),
        deps = if_onednn(if_true = deps, if_false = ["@com_google_googletest//:gtest_main"]),
        # If not building with Graph API, we don't have any tests linked.
        fail_if_no_test_linked = False,
        # If not building with Graph API, we don't have any tests defined either.
        fail_if_no_test_selected = False,
        **kwargs
    )
