# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
