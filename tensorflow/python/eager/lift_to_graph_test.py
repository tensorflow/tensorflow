# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for lift_to_graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import def_function
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager import test
from tensorflow.python.framework import func_graph
from tensorflow.python.ops import resource_variable_ops


class LiftToGraphTest(test.TestCase):

  def testCaptureOrdering(self):
    v1 = resource_variable_ops.ResourceVariable(1.0)
    v2 = resource_variable_ops.ResourceVariable(2.0)
    v3 = resource_variable_ops.ResourceVariable(3.0)

    @def_function.function
    def fn():
      return v1 + v2 + v3

    concrete_fn = fn.get_concrete_function()
    original_captures = concrete_fn.graph.captures
    outputs = concrete_fn.graph.outputs

    for _ in range(100):
      g = func_graph.FuncGraph('lifted')

      lift_to_graph.lift_to_graph(
          outputs, g, add_sources=True, handle_captures=True)
      lifted_captures = g.captures
      self.assertLen(lifted_captures, 3)
      for original_capture, lifted_capture in zip(original_captures.values(),
                                                  lifted_captures.values()):
        self.assertEqual(original_capture.name, lifted_capture.name)


if __name__ == '__main__':
  test.main()
