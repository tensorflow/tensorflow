"""These are the same as Bazel's native cc_libraries."""

# This file is used in OSS only. It is not transformed by copybara. Therefore all paths in this
# file are OSS paths.

# IMPORTANT: Do not remove this load statement. We rely on that //xla/tsl doesn't exist in g3
# to prevent g3 .bzl files from loading this file.
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

_cc_binary = native.cc_binary
_cc_import = native.cc_import
_cc_shared_library = native.cc_shared_library
_cc_test = native.cc_test

cc_binary = _cc_binary
cc_import = _cc_import
cc_shared_library = _cc_shared_library
cc_test = _cc_test

def cc_library(name, deps = None, copts = None, **kwargs):
    """cc_library that works around external issues.

    This rule hides side effects of https://github.com/bazelbuild/bazel/issues/21519,
    and it enables compatibility with a wider set of absl versions and their changes to nullability
    annotation syntax.

    Args:
      name: name of target.
      deps: deps with `xla/tsl:bazel_issue_21519` added.
      copts: copts to which definitions of absl nullability macros are added.
      **kwargs: passed to native.cc_library.
    """

    if deps == None:
        deps = []

    # Horrifying, but needed to prevent a cycle, as `bazel_issue_21519` is an
    # alias of `empty`.
    if name != "empty":
        deps = deps + ["@local_xla//xla/tsl:bazel_issue_21519"]  # buildifier: disable=list-append
        deps = deps + ["@local_tsl//:bazel_issue_21519"]  # buildifier: disable=list-append

    if copts == None:
        copts = []

    # /*absl_nonnull*/, /*absl_nullable*/, and /*absl_nullability_unknown*/ are not yet present in the version
    # of absl we are using.
    # This can be removed when the absl version used is bumped to commit 48f0f91 or newer, likely
    # after July 2025.
    copts = copts + ["-D/*absl_nonnull*/=''", "-D/*absl_nullable*/=''", "-D/*absl_nullability_unknown*/=''"]

    native.cc_library(name = name, deps = deps, copts = copts, **kwargs)
