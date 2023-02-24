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
import numpy as np

from tensorflow.core.function.capture import capture_container
from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import tensor_spec
from tensorflow.python.platform import test


class CaptureContainerTest(test.TestCase, parameterized.TestCase):

  def _prepare_function_captures(self):
    container = capture_container.FunctionCaptures()
    container.capture_by_ref(lambda: 1, "1")
    container.capture_by_ref(lambda: 2, "2")
    return container

  @combinations.generate(combinations.combine(mode=["graph"]))
  def test_capture_by_ref_dict_sz(self):
    container = self._prepare_function_captures()
    self.assertLen(container.by_ref_captures, 2)

  @combinations.generate(combinations.combine(mode=["graph"]))
  def test_capture_by_ref_default_idf(self):
    container = self._prepare_function_captures()
    idf = len(container.by_ref_captures)
    container.capture_by_ref(lambda: 12345)
    capture = container.by_ref_captures[idf]
    lam = capture.external
    self.assertEqual(lam(), 12345)

  @combinations.generate(combinations.combine(mode=["graph"]))
  def test_capture_by_ref_is_by_ref(self):
    container = self._prepare_function_captures()
    capture = container.by_ref_captures["1"]
    self.assertTrue(capture.is_by_ref)

  @combinations.generate(combinations.combine(mode=["graph"]))
  def test_capture_by_ref_with_duplicate_idf(self):
    container = self._prepare_function_captures()
    container.capture_by_ref(lambda: 3, "1")
    self.assertLen(container.by_ref_captures, 2)

  @combinations.generate(combinations.combine(mode=["graph"]))
  def test_get_by_ref_snapshot(self):
    container = self._prepare_function_captures()
    snaptshot = container.get_by_ref_snapshot()
    self.assertDictEqual(snaptshot, {"1": 1, "2": 2})

  @combinations.generate(combinations.combine(mode=["eager",]))
  def test_create_capture_placeholder_eager(self):
    container = self._prepare_function_captures()
    lam = lambda: 12345
    res = container._create_capture_placeholder(lam)
    self.assertEqual(res, 12345)

  @combinations.generate(combinations.combine(mode=["graph",]))
  def test_create_capture_placeholder_graph_tensor(self):
    container = self._prepare_function_captures()
    lam = lambda: constant_op.constant(123)
    spec = tensor_spec.TensorSpec([], np.int32, name="Placeholder:0")
    graph = func_graph.FuncGraph("graph")
    with graph.as_default():
      placeholder = container._create_capture_placeholder(lam)
      self.assertEqual(placeholder.shape, spec.shape)
      self.assertEqual(placeholder.dtype, spec.dtype)
      self.assertEqual(placeholder.name, spec.name)

  @combinations.generate(combinations.combine(mode=["graph",]))
  def test_create_capture_placeholder_graph_nested_tensor(self):
    container = self._prepare_function_captures()
    a = constant_op.constant(1)
    b = constant_op.constant(2.0)
    c = constant_op.constant([1, 2, 3])
    spec_a = tensor_spec.TensorSpec([], np.int32)
    spec_b = tensor_spec.TensorSpec([], np.float32)
    spec_c = tensor_spec.TensorSpec([3,], np.int32)

    value = [{"a": a}, [b, c]]
    lam = lambda: value
    graph = func_graph.FuncGraph("graph")
    with graph.as_default():
      placeholder = container._create_capture_placeholder(lam)
    self.assertLen(placeholder, 2)
    self.assertIn("a", placeholder[0])
    self.assertLen(placeholder[1], 2)

    placeholder_a = placeholder[0]["a"]
    placeholder_b = placeholder[1][0]
    placeholder_c = placeholder[1][1]
    self.assertEqual(placeholder_a.shape, spec_a.shape)
    self.assertEqual(placeholder_a.dtype, spec_a.dtype)
    self.assertEqual(placeholder_b.shape, spec_b.shape)
    self.assertEqual(placeholder_b.dtype, spec_b.dtype)
    self.assertEqual(placeholder_c.shape, spec_c.shape)
    self.assertEqual(placeholder_c.dtype, spec_c.dtype)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
