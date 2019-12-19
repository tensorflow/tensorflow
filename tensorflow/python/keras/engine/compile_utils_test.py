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
"""Tests for compile utitilies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class LossesContainerTest(keras_parameterized.TestCase):

  def test_single_loss(self):
    loss_container = compile_utils.LossesContainer('mse')
    y_t, y_p = array_ops.ones((10, 5)), array_ops.zeros((10, 5))
    total_loss, output_losses = self.evaluate(loss_container(y_t, y_p))

    self.assertTrue(loss_container._built)
    self.assertLen(loss_container._losses, 1)
    self.assertEqual(total_loss, 1.)
    self.assertLen(output_losses, 1)

  def test_loss_list(self):
    loss_container = compile_utils.LossesContainer(['mse', 'mae'], [1, 0.5])

    y_t = [array_ops.ones((10, 1)), array_ops.zeros((10, 1))]
    y_p = [array_ops.ones((10, 1)), array_ops.ones((10, 1))]
    sw = ops.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    total_loss, output_losses = self.evaluate(
        loss_container(y_t, y_p, sample_weight=sw))

    self.assertLen(loss_container._losses, 2)
    self.assertEqual(total_loss, 0.25)
    self.assertLen(output_losses, 2)
    self.assertEqual(output_losses[0], 0)
    self.assertEqual(output_losses[1], 0.25)

  def test_loss_dict(self):
    loss_container = compile_utils.LossesContainer(
        {
            'out1': 'mse',
            'out2': 'mae'
        }, {
            'out1': 1,
            'out2': 0.5
        })

    y_t = {'out1': array_ops.ones((10, 1)), 'out2': array_ops.zeros((10, 1))}
    y_p = {'out1': array_ops.ones((10, 1)), 'out2': array_ops.ones((10, 1))}
    sw = ops.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    total_loss, output_losses = self.evaluate(
        loss_container(y_t, y_p, sample_weight=sw))

    self.assertLen(loss_container._losses, 2)
    self.assertEqual(total_loss, 0.25)
    self.assertLen(output_losses, 2)
    self.assertEqual(output_losses[0], 0)
    self.assertEqual(output_losses[1], 0.25)

  def test_loss_partial_dict_with_output_names(self):
    loss_container = compile_utils.LossesContainer(
        {'out2': 'mae'}, {'out2': 1.}, output_names=['out1', 'out2'])

    y_t = [array_ops.ones((10, 1)), array_ops.zeros((10, 1))]
    y_p = [array_ops.ones((10, 1)), array_ops.ones((10, 1))]
    sw = ops.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    total_loss, output_losses = self.evaluate(
        loss_container(y_t, y_p, sample_weight=sw))

    self.assertEqual(total_loss, 0.5)
    self.assertLen(output_losses, 1)
    self.assertEqual(output_losses[0], 0.5)

  def test_loss_dict_with_nones(self):
    loss_container = compile_utils.LossesContainer({
        'out1': None,
        'out2': 'mae'
    })

    y_t = [array_ops.ones((10, 1)), array_ops.zeros((10, 1))]
    y_p = [array_ops.ones((10, 1)), array_ops.ones((10, 1))]
    sw = ops.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    total_loss, output_losses = self.evaluate(
        loss_container(y_t, y_p, sample_weight=sw))

    self.assertEqual(total_loss, 0.5)
    self.assertLen(output_losses, 1)
    self.assertEqual(output_losses[0], 0.5)

  def test_nested_structure(self):
    loss_container = compile_utils.LossesContainer(
        {
            'b': ['mse', None],
            'a': 'mae'
        }, loss_weights={
            'b': [0.5, 0],
            'a': 1
        })

    y_t = {
        'b': [array_ops.ones((10, 1)),
              array_ops.zeros((10, 1))],
        'a': array_ops.zeros((10, 1))
    }
    y_p = {
        'b': [array_ops.zeros((10, 1)),
              array_ops.zeros((10, 1))],
        'a': array_ops.ones((10, 1))
    }
    sw = ops.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    total_loss, output_losses = self.evaluate(
        loss_container(y_t, y_p, sample_weight=sw))

    self.assertEqual(total_loss, 0.75)
    self.assertLen(output_losses, 2)
    self.assertEqual(output_losses[0], 0.5)
    self.assertEqual(output_losses[1], 0.25)

  def test_broadcast_single_loss(self):
    loss_container = compile_utils.LossesContainer('mse')

    y_t = [array_ops.ones((10, 1)), array_ops.zeros((10, 1))]
    y_p = [array_ops.ones((10, 1)), array_ops.ones((10, 1))]
    sw = ops.convert_to_tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    total_loss, output_losses = self.evaluate(
        loss_container(y_t, y_p, sample_weight=sw))

    self.assertEqual(total_loss, 0.5)
    self.assertLen(output_losses, 2)
    self.assertEqual(output_losses[0], 0.)
    self.assertEqual(output_losses[1], 0.5)


if __name__ == '__main__':
  test.main()
