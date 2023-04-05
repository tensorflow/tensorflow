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

from absl.testing import parameterized

from tensorflow.core.function.capture import capture_container
from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class CachedCaptureDict(test.TestCase, parameterized.TestCase):

  def _prepare_dict(self):
    container_1 = capture_container.CaptureContainer(
        1, constant_op.constant(1), "1")
    container_2 = capture_container.CaptureContainer(
        2, constant_op.constant(2), "2")
    container_3 = capture_container.CaptureContainer(
        3, constant_op.constant(3), "3")

    d = dict({
        "a": container_1,
        "b": container_2,
        "c": container_3})

    capture_d = capture_container.CachedCaptureDict({
        "a": container_1,
        "b": container_2,
        "c": container_3})

    return d, capture_d

  def _compare_capture_container(self, x, y):
    if isinstance(
        x, capture_container.CaptureContainer) and isinstance(
            y, capture_container.CaptureContainer):
      for attr in ["external", "internal", "idf", "is_by_ref"]:
        if getattr(x, attr) != getattr(y, attr):
          return False
      return True
    else:
      return x == y

  @parameterized.parameters(
      ("__contains__", "a"),
      ("__contains__", "not_exist"),
      ("__len__", None),
      ("__getitem__", "a"))
  def test_same_behavior_with_normal_dict(self, method, arg):
    d, capture_d = self._prepare_dict()
    d_method = getattr(d, method)
    capture_d_method = getattr(capture_d, method)
    if arg is None:
      result = self._compare_capture_container(
          d_method(),
          capture_d_method())
    else:
      result = self._compare_capture_container(
          d_method(arg),
          capture_d_method(arg))
    self.assertTrue(result)

  def _extract_tuple_cache_external(self, tpl):
    return [i[0] for i in tpl]

  @parameterized.parameters(
      ("pop",),
      ("__delitem__",))
  def test_pop_and_del(self, method):
    _, capture_d = self._prepare_dict()
    fn = getattr(capture_d, method)
    fn("b")
    cache = capture_d.tuple_cache
    self.assertLen(cache, 2)
    externals = self._extract_tuple_cache_external(cache)
    self.assertSequenceEqual(externals, [1, 3])

  def test_set_item(self):
    _, capture_d = self._prepare_dict()
    container_4 = capture_container.CaptureContainer(
        4, constant_op.constant(4), "4")
    capture_d["d"] = container_4
    cache = capture_d.tuple_cache
    self.assertLen(cache, 4)
    externals = self._extract_tuple_cache_external(cache)
    self.assertSequenceEqual(externals, [1, 2, 3, 4])

  def test_tuple_cache(self):
    _, capture_d = self._prepare_dict()
    cache = capture_d.tuple_cache
    for ele in cache:
      self.assertLen(ele, 2)
    externals = self._extract_tuple_cache_external(cache)
    self.assertSequenceEqual(externals, [1, 2, 3])


class CaptureContainerTest(test.TestCase, parameterized.TestCase):

  def _prepare_function_captures(self):
    container = capture_container.FunctionCaptures()
    graph = ops.get_default_graph()
    container._capture_by_ref(graph, lambda: 1, "1")
    container._capture_by_ref(graph, lambda: 2, "2")
    return container

  @combinations.generate(combinations.combine(mode=["graph"]))
  def test__capture_by_ref_dict_sz(self):
    container = self._prepare_function_captures()
    self.assertLen(container.by_ref_captures, 2)

  @combinations.generate(combinations.combine(mode=["graph"]))
  def test__capture_by_ref_default_idf(self):
    container = self._prepare_function_captures()
    graph = ops.get_default_graph()
    idf = len(container.by_ref_captures)
    container._capture_by_ref(graph, lambda: 12345)
    capture = container.by_ref_captures[idf]
    lam = capture.lambda_fn
    self.assertEqual(lam(), [])

  @combinations.generate(combinations.combine(mode=["graph"]))
  def test__capture_by_ref_with_duplicate_idf(self):
    container = self._prepare_function_captures()
    graph = ops.get_default_graph()
    container._capture_by_ref(graph, lambda: 3, "1")
    self.assertLen(container.by_ref_captures, 2)

  @combinations.generate(combinations.combine(mode=["graph"]))
  def test_get_by_ref_snapshot(self):
    container = self._prepare_function_captures()
    snaptshot = container.get_by_ref_snapshot()
    self.assertDictEqual(snaptshot, {"1": [], "2": []})


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
