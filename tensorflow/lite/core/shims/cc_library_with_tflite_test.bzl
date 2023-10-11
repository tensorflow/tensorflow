"""Unit tests for cc_library_with_tflite."""

load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load(":cc_library_with_tflite.bzl", "add_suffix")

def _add_suffix_test(ctx):
    """Unit test for add_suffix."""
    env = unittest.begin(ctx)

    asserts.equals(env, "@repo//path/to:foo_test", add_suffix("@repo//path/to:foo", "_test"))
    asserts.equals(env, "@repo//path/to/foo:foo_test", add_suffix("@repo//path/to/foo", "_test"))
    asserts.equals(env, "@//path/to:foo_test", add_suffix("@//path/to:foo", "_test"))
    asserts.equals(env, "@//path/to/foo:foo_test", add_suffix("@//path/to/foo", "_test"))
    asserts.equals(env, "//path/to:foo_test", add_suffix("//path/to:foo", "_test"))
    asserts.equals(env, "//path/to/foo:foo_test", add_suffix("//path/to/foo", "_test"))
    asserts.equals(env, "foo_test", add_suffix("foo", "_test"))
    asserts.equals(env, ":foo_test", add_suffix(":foo", "_test"))

    return unittest.end(env)

add_suffix_test = unittest.make(_add_suffix_test)

def cc_library_with_tflite_test_suite(name):
    """Creates the test targets and test suite for some tests of add_suffix."""
    unittest.suite(
        name,
        add_suffix_test,
    )
