# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SavedModel simple load functionality."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import simple_save
from tensorflow.python.saved_model import simple_load


class SimpleLoadTest(test.TestCase):

  def _init_and_validate_variable(self, sess, variable_name, variable_value):
    v = variables.Variable(variable_value, name=variable_name)
    sess.run(variables.global_variables_initializer())
    self.assertEqual(variable_value, v.eval())
    return v

  def _check_variable_info(self, actual_variable, expected_variable):
    self.assertEqual(actual_variable.name, expected_variable.name)
    self.assertEqual(actual_variable.dtype, expected_variable.dtype)
    self.assertEqual(len(actual_variable.shape), len(expected_variable.shape))
    for i in range(len(actual_variable.shape)):
      self.assertEqual(actual_variable.shape[i], expected_variable.shape[i])

  def _check_tensor(self, actual_tensor, expected_tensor):
    self.assertEqual(actual_tensor.name, expected_tensor.name)
    self.assertEqual(actual_tensor.dtype, expected_tensor.dtype)
    self.assertEqual(
        len(actual_tensor.shape), len(expected_tensor.shape))
    for i in range(len(actual_tensor.shape)):
      self.assertEqual(actual_tensor.shape[i], expected_tensor.shape[i])

  def testSimpleLoad(self):
    """Test simple_load that uses the simple_save default parameters."""
    export_dir = os.path.join(test.get_temp_dir(), "test_simple_load")

    # Initialize input and output variables and save a prediction graph using
    # the default parameters.
    with self.test_session(graph=ops.Graph()) as sess:
      var_x = self._init_and_validate_variable(sess, "var_x", 1)
      var_y = self._init_and_validate_variable(sess, "var_y", 2)
      inputs = {"x": var_x}
      outputs = {"y": var_y}
      simple_save.simple_save(sess, export_dir, inputs, outputs)

    # Restore the graph with simple load function and check inputs/outputs
    with self.test_session(graph=ops.Graph()) as sess:
      inputs, outputs = simple_load.simple_load(sess, export_dir)

      # Metadata of loaded inputs and outputs
      self.assertEqual(len(inputs), 1)
      self.assertEqual(len(outputs), 1)
      self.assertIn("x", inputs)
      self.assertIn("y", outputs)

      # Check value of loaded inputs and outputs
      loaded_x = inputs["x"]
      loaded_y = outputs["y"]
      self.assertEqual(1, loaded_x.eval())
      self.assertEqual(2, loaded_y.eval())

      # Check tensor information
      self._check_tensor(var_x, loaded_x)
      self._check_tensor(var_y, loaded_y)


if __name__ == "__main__":
  test.main()
