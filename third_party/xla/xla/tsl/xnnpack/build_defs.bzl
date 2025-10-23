"""Macros for XNNPACK and YNNPACK."""

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def if_ynnpack(if_true, if_false = []):
    """Selection based on whether we are building XLA with YNNPACK integration.

    Args:
      if_true: Expression to evaluate if building with YNNPACK.
      if_false: Expression to evaluate if building without YNNPACK.

    Returns:
      A select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        # YNNPACK is not tested on Windows.
        "//xla/tsl:windows": if_false,
        "//conditions:default": if_true,
    })

def ynn_cc_test(
        srcs = [],
        deps = [],
        **kwargs):
    """xla_cc_test rule with empty src and deps if not building with YNNPACK."""
    xla_cc_test(
        # CC_TEST_OK=Just defining `xla_cc_test` rule to be used in XLA.
        srcs = if_ynnpack(srcs),
        deps = if_ynnpack(if_true = deps, if_false = ["@com_google_googletest//:gtest_main"]),
        # If not building with YNNPACK, we don't have any tests linked.
        fail_if_no_test_linked = False,
        # If not building with YNNPACK, we don't have any tests defined either.
        fail_if_no_test_selected = False,
        **kwargs
    )
