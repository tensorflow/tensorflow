"""Provides an indirection layer to bazel cc_rules"""

load(
    "//xla/tsl:package_groups.bzl",
    "DEFAULT_LOAD_VISIBILITY",
    "LEGACY_TSL_PLATFORM_RULES_CC_USERS",
)
load(
    "//xla/tsl/platform/default:rules_cc.bzl",
    _cc_binary = "cc_binary",
    _cc_import = "cc_import",
    _cc_library = "cc_library",
    _cc_shared_library = "cc_shared_library",
    _cc_test = "cc_test",
)

visibility(DEFAULT_LOAD_VISIBILITY + LEGACY_TSL_PLATFORM_RULES_CC_USERS)

cc_binary = _cc_binary
cc_import = _cc_import
cc_library = _cc_library
cc_shared_library = _cc_shared_library
cc_test = _cc_test

def cc_library_optional_no_mkl(name, deps = [], mkl_deps = [], **kwargs):
    """Defines two `cc_library` rules, one with MKL and one without.

    Args:
      name: The name of the library.
      deps: The dependencies of the library.
      mkl_deps: The dependencies that depend on `eigen_contraction_kernel`, or the target itself.
      **kwargs: Other arguments to pass to the `cc_library` rule.
    """
    cc_library(
        name = name,
        deps = deps + mkl_deps,
        **kwargs
    )
    cc_library(
        name = name + "_no_mkl",
        deps = deps + [mkl_dep + "_no_mkl" for mkl_dep in mkl_deps],
        **kwargs
    )
