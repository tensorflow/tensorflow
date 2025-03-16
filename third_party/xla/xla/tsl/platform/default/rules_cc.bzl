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

def cc_library(name, deps = None, **kwargs):
    """cc_library that hides side effects of https://github.com/bazelbuild/bazel/issues/21519.

    Args:
      name: name of target.
      deps: deps with `xla/tsl:bazel_issue_21519` added.
      **kwargs: passed to native.cc_library.
    """

    if deps == None:
        deps = []

    # Horrifying, but needed to prevent a cycle, as `bazel_issue_21519` is an
    # alias of `empty`.
    if name != "empty":
        deps = deps + ["@local_xla//xla/tsl:bazel_issue_21519"]  # buildifier: disable=list-append
        deps = deps + ["@local_tsl//:bazel_issue_21519"]  # buildifier: disable=list-append

    native.cc_library(name = name, deps = deps, **kwargs)
