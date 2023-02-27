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
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class CaptureContainerTest(test.TestCase, parameterized.TestCase):

  def _prepare_function_captures(self):
    container = capture_container.FunctionCaptures()
    graph = ops.get_default_graph()
    container.capture_by_ref(graph, lambda: 1, "1")
    container.capture_by_ref(graph, lambda: 2, "2")
    return container

  @combinations.generate(combinations.combine(mode=["graph"]))
  def test_capture_by_ref_dict_sz(self):
    container = self._prepare_function_captures()
    self.assertLen(container.by_ref_captures, 2)

  @combinations.generate(combinations.combine(mode=["graph"]))
  def test_capture_by_ref_default_idf(self):
    container = self._prepare_function_captures()
    graph = ops.get_default_graph()
    idf = len(container.by_ref_captures)
    container.capture_by_ref(graph, lambda: 12345)
    capture = container.by_ref_captures[idf]
    lam = capture.lambda_fn
    self.assertEqual(lam(), 12345)

  @combinations.generate(combinations.combine(mode=["graph"]))
  def test_capture_by_ref_with_duplicate_idf(self):
    container = self._prepare_function_captures()
    graph = ops.get_default_graph()
    container.capture_by_ref(graph, lambda: 3, "1")
    self.assertLen(container.by_ref_captures, 2)

  @combinations.generate(combinations.combine(mode=["graph"]))
  def test_get_by_ref_snapshot(self):
    container = self._prepare_function_captures()
    snaptshot = container.get_by_ref_snapshot()
    self.assertDictEqual(snaptshot, {"1": 1, "2": 2})


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
