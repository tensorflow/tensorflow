# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for training utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.platform import test


class ModelInputsTest(test.TestCase):

  def test_single_thing(self):
    a = np.ones(10)
    model_inputs = training_utils.ModelInputs(a)
    self.assertEqual(['input_1'], model_inputs.get_input_names())
    vals = model_inputs.get_symbolic_inputs()
    self.assertTrue(tensor_util.is_tensor(vals))
    vals = model_inputs.get_symbolic_inputs(return_single_as_list=True)
    self.assertEqual(1, len(vals))
    self.assertTrue(tensor_util.is_tensor(vals[0]))

  def test_single_thing_eager(self):
    with context.eager_mode():
      a = np.ones(10)
      model_inputs = training_utils.ModelInputs(a)
      self.assertEqual(['input_1'], model_inputs.get_input_names())
      val = model_inputs.get_symbolic_inputs()
      self.assertTrue(tf_utils.is_symbolic_tensor(val))
      vals = model_inputs.get_symbolic_inputs(return_single_as_list=True)
      self.assertEqual(1, len(vals))
      self.assertTrue(tf_utils.is_symbolic_tensor(vals[0]))

  def test_list(self):
    a = [np.ones(10), np.ones(20)]
    model_inputs = training_utils.ModelInputs(a)
    self.assertEqual(['input_1', 'input_2'], model_inputs.get_input_names())
    vals = model_inputs.get_symbolic_inputs()
    self.assertTrue(tensor_util.is_tensor(vals[0]))
    self.assertTrue(tensor_util.is_tensor(vals[1]))

  def test_list_eager(self):
    with context.eager_mode():
      a = [np.ones(10), np.ones(20)]
      model_inputs = training_utils.ModelInputs(a)
      self.assertEqual(['input_1', 'input_2'], model_inputs.get_input_names())
      vals = model_inputs.get_symbolic_inputs()
      self.assertTrue(tf_utils.is_symbolic_tensor(vals[0]))
      self.assertTrue(tf_utils.is_symbolic_tensor(vals[1]))

  def test_dict(self):
    a = {'b': np.ones(10), 'a': np.ones(20)}
    model_inputs = training_utils.ModelInputs(a)
    self.assertEqual(['a', 'b'], model_inputs.get_input_names())
    vals = model_inputs.get_symbolic_inputs()
    self.assertTrue(tensor_util.is_tensor(vals['a']))
    self.assertTrue(tensor_util.is_tensor(vals['b']))

  def test_dict_eager(self):
    with context.eager_mode():
      a = {'b': np.ones(10), 'a': np.ones(20)}
      model_inputs = training_utils.ModelInputs(a)
      self.assertEqual(['a', 'b'], model_inputs.get_input_names())
      vals = model_inputs.get_symbolic_inputs()
      self.assertTrue(tf_utils.is_symbolic_tensor(vals['a']))
      self.assertTrue(tf_utils.is_symbolic_tensor(vals['b']))


if __name__ == '__main__':
  test.main()
