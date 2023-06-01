# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.function capture containers."""
import copy

from absl.testing import parameterized

from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.python.platform import test


class MutationAwareDictTest(test.TestCase, parameterized.TestCase):

  def _prepare_dict(self):
    d = {"a": 1, "b": 2, "c": 3}
    mutation_d = capture_container.MutationAwareDict(copy.copy(d))
    return d, mutation_d

  @parameterized.parameters(
      ("__contains__", "a"),
      ("__contains__", "not_exist"),
      ("__len__", None),
      ("__getitem__", "a"))
  def test_same_behavior_with_normal_dict(self, method, arg):
    d, mutation_d = self._prepare_dict()
    d_method = getattr(d, method)
    mutation_d_method = getattr(mutation_d, method)
    if arg is None:
      self.assertEqual(d_method(), mutation_d_method())
    else:
      self.assertEqual(d_method(arg), mutation_d_method(arg))

  @parameterized.parameters(
      ("pop",),
      ("__delitem__",))
  def test_pop_and_del(self, method):
    d, mutation_d = self._prepare_dict()
    d_method = getattr(d, method)
    mutation_d_method = getattr(mutation_d, method)
    d_method("b")
    mutation_d_method("b")
    self.assertListEqual(list(d.keys()), list(mutation_d.keys()))
    self.assertListEqual(list(d.values()), list(mutation_d.values()))

  def test_mutatation_ops(self):
    _, d = self._prepare_dict()
    with self.subTest("set"):
      d["d"] = 4
      self.assertTrue(d.mutated)
    with self.subTest("pop"):
      d.pop("d")
      self.assertTrue(d.mutated)
    with self.subTest("del"):
      del d["c"]
      self.assertTrue(d.mutated)
    with self.subTest("clear"):
      d.clear()
      self.assertTrue(d.mutated)

  def test_mutated_property(self):
    _, d = self._prepare_dict()
    with self.subTest("initial_state"):
      self.assertTrue(d.mutated)
    with self.subTest("setter"):
      d.mutated = False
      self.assertFalse(d.mutated)


class FunctionCapturesTest(test.TestCase, parameterized.TestCase):

  def test_add_or_replace(self):
    fn_captures = capture_container.FunctionCaptures()
    fn_captures.add_or_replace("a", 1, -1, is_by_ref=False)
    fn_captures.add_or_replace("aa", 1, -1, 0, is_by_ref=True)

    with self.subTest("add_by_val"):
      self.assertLen(fn_captures.by_val_internal, 1)
      self.assertLen(fn_captures.by_val_external, 1)

    with self.subTest("add_by_ref"):
      self.assertLen(fn_captures.by_ref_internal, 1)
      self.assertLen(fn_captures.by_ref_external, 1)
      self.assertLen(fn_captures.by_ref_tracetype, 1)

    fn_captures.add_or_replace("a", 2, -2, is_by_ref=False)
    with self.subTest("replace_by_val"):
      self.assertLen(fn_captures.by_val_internal, 1)
      self.assertLen(fn_captures.by_val_external, 1)
      self.assertEqual(fn_captures.by_val_external["a"], 2)
      self.assertEqual(fn_captures.by_val_internal["a"], -2)

  def test_by_val_capture_tuples(self):
    fn_captures = capture_container.FunctionCaptures()

    with self.subTest("initial_state"):
      self.assertEmpty(fn_captures.by_val_capture_tuples)

    with self.subTest("add"):
      fn_captures.add_or_replace("a", 1, -1, is_by_ref=False)
      self.assertLen(fn_captures.by_val_capture_tuples, 1)
      self.assertSequenceEqual(
          fn_captures.by_val_capture_tuples,
          ((1, -1),))

      fn_captures.add_or_replace("b", 2, -2, is_by_ref=False)
      self.assertLen(fn_captures.by_val_capture_tuples, 2)
      self.assertSequenceEqual(
          fn_captures.by_val_capture_tuples,
          ((1, -1), (2, -2)))

    with self.subTest("replace"):
      fn_captures.add_or_replace("a", 1, -3, is_by_ref=False)
      self.assertLen(fn_captures.by_val_capture_tuples, 2)
      self.assertSequenceEqual(
          fn_captures.by_val_capture_tuples,
          ((1, -3), (2, -2)))

    with self.subTest("pop"):
      fn_captures.pop("b", is_by_ref=False)
      self.assertSequenceEqual(
          fn_captures.by_val_capture_tuples,
          ((1, -3),))

    with self.subTest("reset"):
      fn_captures.reset_captures([10, 20], [-10, -20])
      self.assertSequenceEqual(
          fn_captures.by_val_capture_tuples,
          ((10, -10), (20, -20)))

    with self.subTest("clear"):
      fn_captures.clear()
      self.assertEmpty(fn_captures.by_val_capture_tuples)

  def test_capture_types(self):
    class FakePlaceholder():
      pass

    fn_captures = capture_container.FunctionCaptures()
    fn_captures.add_or_replace("v1", 1, FakePlaceholder(), is_by_ref=False)
    fn_captures.add_or_replace("v2", 2, FakePlaceholder(), is_by_ref=False)
    fn_captures.add_or_replace("v3", 3, FakePlaceholder(), is_by_ref=False)
    fn_captures.add_or_replace(
        "r1", 1, FakePlaceholder(), trace_type.from_value(4), is_by_ref=True)
    fn_captures.add_or_replace(
        "r2", 2, FakePlaceholder(), trace_type.from_value(5), is_by_ref=True)

    self.assertLen(fn_captures.capture_types, 5)
    self.assertEqual(fn_captures.capture_types["v1"], trace_type.from_value(1))
    self.assertEqual(fn_captures.capture_types["v2"], trace_type.from_value(2))
    self.assertEqual(fn_captures.capture_types["v3"], trace_type.from_value(3))
    self.assertEqual(fn_captures.capture_types["r1"], trace_type.from_value(4))
    self.assertEqual(fn_captures.capture_types["r2"], trace_type.from_value(5))


if __name__ == "__main__":
  test.main()
