"""Reexports googletest_deps from upstream googletest."""

# protobuf loads for @com_google_googletest//:googletest_deps.bzl so we need to
# provide one in the wrapper.
load("@com_google_googletest_upstream//:googletest_deps.bzl", upstream_deps = "googletest_deps")

def googletest_deps():
    upstream_deps()
