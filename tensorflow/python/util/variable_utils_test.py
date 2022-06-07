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
# =============================================================================
"""Tests for variable_utils."""

from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils


class CT(composite_tensor.CompositeTensor):
  """A generic CompositeTensor, used for constructing tests."""

  @property
  def _type_spec(self):
    pass


class CT2(composite_tensor.CompositeTensor):
  """Another CompositeTensor, used for constructing tests."""

  def __init__(self, component):
    self.component = component

  @property
  def _type_spec(self):
    pass

  def _convert_variables_to_tensors(self):
    return CT2(ops.convert_to_tensor(self.component))


@test_util.run_all_in_graph_and_eager_modes
class VariableUtilsTest(test.TestCase):

  def test_convert_variables_to_tensors(self):
    ct = CT()
    data = [resource_variable_ops.ResourceVariable(1),
            resource_variable_ops.ResourceVariable(2),
            constant_op.constant(3),
            [4],
            5,
            ct]
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())

    results = variable_utils.convert_variables_to_tensors(data)
    expected_results = [1, 2, 3, [4], 5, ct]
    # Only ResourceVariables are converted to Tensors.
    self.assertIsInstance(results[0], ops.Tensor)
    self.assertIsInstance(results[1], ops.Tensor)
    self.assertIsInstance(results[2], ops.Tensor)
    self.assertIsInstance(results[3], list)
    self.assertIsInstance(results[4], int)
    self.assertIs(results[5], ct)
    results[:3] = self.evaluate(results[:3])
    self.assertAllEqual(results, expected_results)

  def test_convert_variables_in_composite_tensor(self):
    ct2 = CT2(resource_variable_ops.ResourceVariable(1))
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())

    self.assertIsInstance(ct2.component,
                          resource_variable_ops.ResourceVariable)
    result = variable_utils.convert_variables_to_tensors(ct2)
    self.assertIsInstance(result.component, ops.Tensor)
    self.assertAllEqual(result.component, 1)

  def test_replace_variables_with_atoms(self):
    data = [resource_variable_ops.ResourceVariable(1),
            resource_variable_ops.ResourceVariable(2),
            constant_op.constant(3),
            [4],
            5]
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())

    results = variable_utils.replace_variables_with_atoms(data)
    expected_results = [0, 0, 3, [4], 5]
    # Only ResourceVariables are replaced with int 0s.
    self.assertIsInstance(results[0], int)
    self.assertIsInstance(results[1], int)
    self.assertIsInstance(results[2], ops.Tensor)
    self.assertIsInstance(results[3], list)
    self.assertIsInstance(results[4], int)
    results[2] = self.evaluate(results[2])
    self.assertAllEqual(results, expected_results)

    # Make sure 0 is a tf.nest atom with expand_composites=True.
    flat_results = nest.flatten(results, expand_composites=True)
    expected_flat_results = [0, 0, 3, 4, 5]
    self.assertAllEqual(flat_results, expected_flat_results)


if __name__ == "__main__":
  test.main()
