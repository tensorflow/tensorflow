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

from tensorflow.python.eager import def_function
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat


class LiftToGraphTest(test.TestCase):

  def testCaptureOrdering(self):
    v1 = resource_variable_ops.ResourceVariable(1.0)
    v2 = resource_variable_ops.ResourceVariable(2.0)
    v3 = resource_variable_ops.ResourceVariable(3.0)

    @def_function.function
    def fn():
      return v1 + v2 + v3

    concrete_fn = fn.get_concrete_function()
    original_captures = concrete_fn.graph.internal_captures
    outputs = concrete_fn.graph.outputs

    for _ in range(100):
      g = func_graph.FuncGraph('lifted')

      lift_to_graph.lift_to_graph(
          outputs, g, add_sources=True, handle_captures=True)
      lifted_captures = g.internal_captures
      self.assertLen(lifted_captures, 3)
      for original, lifted in zip(original_captures, lifted_captures):
        self.assertEqual(original.name, lifted.name)

  def testClassAttrsRemoved(self):
    """Tests that _class attrs (from colocate_with()) are removed."""
    @def_function.function
    def fn():
      two = constant_op.constant(2.0, name='two')
      ten = constant_op.constant(10.0, name='ten')
      twenty = math_ops.multiply(two, ten, name='twenty')
      three = constant_op.constant(3.0, name='three')
      with framework_ops.colocate_with(twenty):
        thirty = math_ops.multiply(three, ten, name='thirty')
      return ten, twenty, thirty

    concrete_fn = fn.get_concrete_function()
    self.assertItemsEqual(  # Before lifting, 'fn' has colocation attrs.
        concrete_fn.graph.get_operation_by_name('thirty').colocation_groups(),
        [compat.as_bytes('loc:@twenty')])
    thirty_out = concrete_fn.graph.outputs[2]

    g = func_graph.FuncGraph('lifted')
    lift_to_graph.lift_to_graph([thirty_out], g)

    # After lifting, colocation attrs are gone.
    ops = g.get_operations()
    self.assertItemsEqual([op.name for op in ops],
                          ['three', 'ten', 'thirty',  # Lifted from `fn` body.
                           thirty_out.op.name])  # Wrapper for output.
    for op in ops:
      with self.assertRaises(ValueError):
        class_attr = op.get_attr('_class')  # Expected not to exist.
        print('Unexpected class_attr', class_attr, 'on', op.name)
      self.assertItemsEqual(op.colocation_groups(),  # Expect default self-ref.
                            [compat.as_bytes('loc:@%s' % op.name)])


if __name__ == '__main__':
  test.main()
