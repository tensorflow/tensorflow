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
from tensorflow.python.keras import metrics as metrics_mod
from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class LossesContainerTest(keras_parameterized.TestCase):

  def test_single_loss(self):
    loss_container = compile_utils.LossesContainer('mse')
    y_t, y_p = array_ops.ones((10, 5)), array_ops.zeros((10, 5))
    total_loss = loss_container(y_t, y_p)

    self.assertTrue(loss_container._built)
    self.assertLen(loss_container._losses, 1)
    self.assertEqual(total_loss.numpy(), 1.)
    self.assertLen(loss_container.metrics, 1)

    loss_metric = loss_container.metrics[0]
    self.assertEqual(loss_metric.name, 'loss')
    self.assertEqual(loss_metric.result().numpy(), 1.)

  def test_loss_list(self):
    loss_container = compile_utils.LossesContainer(['mse', 'mae'], [1, 0.5])

    y_t = [array_ops.ones((10, 1)), array_ops.zeros((10, 1))]
    y_p = [array_ops.ones((10, 1)), array_ops.ones((10, 1))]
    sw = ops.convert_to_tensor_v2([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    total_loss = loss_container(y_t, y_p, sample_weight=sw)

    self.assertEqual(loss_container._output_names, ['output_1', 'output_2'])

    self.assertLen(loss_container._losses, 2)
    self.assertEqual(total_loss.numpy(), 0.25)

    loss_metric = loss_container.metrics[0]
    self.assertEqual(loss_metric.name, 'loss')
    self.assertEqual(loss_metric.result().numpy(), 0.25)

    output_1_metric = loss_container.metrics[1]
    self.assertEqual(output_1_metric.name, 'output_1_loss')
    self.assertEqual(output_1_metric.result().numpy(), 0)

    output_2_metric = loss_container.metrics[2]
    self.assertEqual(output_2_metric.name, 'output_2_loss')
    self.assertEqual(output_2_metric.result().numpy(), 0.5)

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
    sw = ops.convert_to_tensor_v2([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    total_loss = loss_container(y_t, y_p, sample_weight=sw)

    self.assertLen(loss_container._losses, 2)
    self.assertEqual(total_loss.numpy(), 0.25)
    self.assertLen(loss_container.metrics, 3)

    loss_metric = loss_container.metrics[0]
    self.assertEqual(loss_metric.name, 'loss')
    self.assertEqual(loss_metric.result().numpy(), 0.25)

    out1_metric = loss_container.metrics[1]
    self.assertEqual(out1_metric.name, 'out1_loss')
    self.assertEqual(out1_metric.result().numpy(), 0)

    out2_metric = loss_container.metrics[2]
    self.assertEqual(out2_metric.name, 'out2_loss')
    self.assertEqual(out2_metric.result().numpy(), 0.5)

  def test_loss_partial_dict_with_output_names(self):
    loss_container = compile_utils.LossesContainer(
        {'out2': 'mae'}, {'out2': 1.}, output_names=['out1', 'out2'])

    y_t = [array_ops.ones((10, 1)), array_ops.zeros((10, 1))]
    y_p = [array_ops.ones((10, 1)), array_ops.ones((10, 1))]
    sw = ops.convert_to_tensor_v2([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    total_loss = loss_container(y_t, y_p, sample_weight=sw)

    self.assertEqual(total_loss.numpy(), 0.5)
    self.assertLen(loss_container.metrics, 2)

    loss_metric = loss_container.metrics[0]
    self.assertEqual(loss_metric.name, 'loss')
    self.assertEqual(loss_metric.result().numpy(), 0.5)

    out2_metric = loss_container.metrics[1]
    self.assertEqual(out2_metric.name, 'out2_loss')
    self.assertEqual(out2_metric.result().numpy(), 0.5)

  def test_loss_dict_with_nones(self):
    loss_container = compile_utils.LossesContainer({
        'out1': None,
        'out2': 'mae'
    })

    y_t = {'out1': array_ops.ones((10, 1)), 'out2': array_ops.zeros((10, 1))}
    y_p = {'out1': array_ops.ones((10, 1)), 'out2': array_ops.ones((10, 1))}
    sw = ops.convert_to_tensor_v2([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    total_loss = loss_container(y_t, y_p, sample_weight=sw)

    self.assertEqual(total_loss.numpy(), 0.5)
    self.assertLen(loss_container.metrics, 2)

    loss_metric = loss_container.metrics[0]
    self.assertEqual(loss_metric.name, 'loss')
    self.assertEqual(loss_metric.result().numpy(), 0.5)

    out2_metric = loss_container.metrics[1]
    self.assertEqual(out2_metric.name, 'out2_loss')
    self.assertEqual(out2_metric.result().numpy(), 0.5)

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
    sw = ops.convert_to_tensor_v2([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    total_loss = loss_container(y_t, y_p, sample_weight=sw)
    self.assertEqual(total_loss.numpy(), 0.75)
    self.assertLen(loss_container.metrics, 3)

    loss_metric = loss_container.metrics[0]
    self.assertEqual(loss_metric.name, 'loss')
    self.assertEqual(loss_metric.result().numpy(), 0.75)

    a_metric = loss_container.metrics[1]
    self.assertEqual(a_metric.name, 'a_loss')
    self.assertEqual(a_metric.result().numpy(), 0.5)

    b_1_metric = loss_container.metrics[2]
    self.assertEqual(b_1_metric.name, 'b_1_loss')
    self.assertEqual(b_1_metric.result().numpy(), 0.5)

  def test_broadcast_single_loss(self):
    loss_container = compile_utils.LossesContainer('mse')

    y_t = [array_ops.ones((10, 1)), array_ops.zeros((10, 1))]
    y_p = [array_ops.ones((10, 1)), array_ops.ones((10, 1))]
    sw = ops.convert_to_tensor_v2([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    total_loss = loss_container(y_t, y_p, sample_weight=sw)
    self.assertEqual(total_loss.numpy(), 0.5)
    self.assertLen(loss_container.metrics, 3)

    loss_metric = loss_container.metrics[0]
    self.assertEqual(loss_metric.name, 'loss')
    self.assertEqual(loss_metric.result().numpy(), 0.5)

    output_1_metric = loss_container.metrics[1]
    self.assertEqual(output_1_metric.name, 'output_1_loss')
    self.assertEqual(output_1_metric.result().numpy(), 0.)

    output_2_metric = loss_container.metrics[2]
    self.assertEqual(output_2_metric.name, 'output_2_loss')
    self.assertEqual(output_2_metric.result().numpy(), 0.5)

  def test_missing_label_with_no_loss(self):
    # It's ok to exclude a label if that label has no
    # losses or metrics associated with it.
    loss_container = compile_utils.LossesContainer({
        'output1': 'mse',
        'output3': 'mae'
    })

    y_p = {
        'output1': ops.convert_to_tensor([[0], [1], [2]]),
        'output2': ops.convert_to_tensor([[3], [4], [5]]),
        'output3': ops.convert_to_tensor([[6], [7], [8]])
    }
    y_t = {
        'output1': ops.convert_to_tensor([[1], [2], [3]]),
        'output3': ops.convert_to_tensor([[4], [5], [6]])
    }

    total_loss = loss_container(y_t, y_p)
    self.assertEqual(total_loss.numpy(), 3.)
    self.assertLen(loss_container.metrics, 3)

    loss_metric = loss_container.metrics[0]
    self.assertEqual(loss_metric.name, 'loss')
    self.assertEqual(loss_metric.result().numpy(), 3.)

    output_1_metric = loss_container.metrics[1]
    self.assertEqual(output_1_metric.name, 'output1_loss')
    self.assertEqual(output_1_metric.result().numpy(), 1.)

    output_3_metric = loss_container.metrics[2]
    self.assertEqual(output_3_metric.name, 'output3_loss')
    self.assertEqual(output_3_metric.result().numpy(), 2.)


class MetricsContainerTest(keras_parameterized.TestCase):

  def test_single_metric(self):
    metric_container = compile_utils.MetricsContainer('mse')
    y_t, y_p = array_ops.ones((10, 5)), array_ops.zeros((10, 5))
    metric_container.update_state(y_t, y_p)

    self.assertLen(metric_container.metrics, 1)
    metric = metric_container.metrics[0]
    self.assertEqual(metric.name, 'mse')
    self.assertEqual(metric.result().numpy(), 1.)

  def test_list_of_metrics_one_output(self):
    metric_container = compile_utils.MetricsContainer(['mse', 'mae'])
    y_t, y_p = 2 * array_ops.ones((10, 5)), array_ops.zeros((10, 5))
    metric_container.update_state(y_t, y_p)
    self.assertLen(metric_container.metrics, 2)

    mse_metric = metric_container.metrics[0]
    self.assertEqual(mse_metric.name, 'mse')
    self.assertEqual(mse_metric.result().numpy(), 4.)

    mae_metric = metric_container.metrics[1]
    self.assertEqual(mae_metric.name, 'mae')
    self.assertEqual(mae_metric.result().numpy(), 2.)

  def test_list_of_metrics_list_of_outputs(self):
    metric_container = compile_utils.MetricsContainer(
        metrics=['mse', 'mae'],  # Should broadcast to both outputs.
        weighted_metrics=['accuracy'])  # Should broadcast to both outputs.

    y_t = [array_ops.ones((10, 1)), array_ops.zeros((10, 1))]
    y_p = [array_ops.ones((10, 1)), 2 * array_ops.ones((10, 1))]
    sw = ops.convert_to_tensor_v2([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    metric_container.update_state(y_t, y_p, sample_weight=sw)
    self.assertLen(metric_container.metrics, 6)

    mse_metric = metric_container.metrics[0]
    self.assertEqual(mse_metric.name, 'output_1_mse')
    self.assertEqual(mse_metric.result().numpy(), 0.)

    mse_metric = metric_container.metrics[1]
    self.assertEqual(mse_metric.name, 'output_1_mae')
    self.assertEqual(mse_metric.result().numpy(), 0.)

    acc_metric_1 = metric_container.metrics[2]
    self.assertEqual(acc_metric_1.name, 'output_1_accuracy')
    self.assertEqual(acc_metric_1.result().numpy(), 1.)
    self.assertEqual(acc_metric_1._fn, metrics_mod.binary_accuracy)

    mae_metric = metric_container.metrics[3]
    self.assertEqual(mae_metric.name, 'output_2_mse')
    self.assertEqual(mae_metric.result().numpy(), 4.)

    mae_metric = metric_container.metrics[4]
    self.assertEqual(mae_metric.name, 'output_2_mae')
    self.assertEqual(mae_metric.result().numpy(), 2.)

    acc_metric_2 = metric_container.metrics[5]
    self.assertEqual(acc_metric_2.name, 'output_2_accuracy')
    self.assertEqual(acc_metric_2.result().numpy(), 0.)
    self.assertEqual(acc_metric_2._fn, metrics_mod.binary_accuracy)

  def test_metric_dict(self):
    metric_container = compile_utils.MetricsContainer(
        metrics={
            'out1': 'mse',
            'out2': 'mae'
        },
        weighted_metrics={
            'out1': 'mse',
            'out2': 'mae'
        })

    y_t = {'out1': array_ops.ones((10, 1)), 'out2': array_ops.zeros((10, 1))}
    y_p = {'out1': array_ops.ones((10, 1)), 'out2': 2 * array_ops.ones((10, 1))}
    sw = ops.convert_to_tensor_v2([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    metric_container.update_state(y_t, y_p, sample_weight=sw)

    mse_metric = metric_container.metrics[0]
    self.assertEqual(mse_metric.name, 'out1_mse')
    self.assertEqual(mse_metric.result().numpy(), 0.)

    weighted_mse_metric = metric_container.metrics[1]
    self.assertEqual(weighted_mse_metric.name, 'out1_weighted_mse')
    self.assertEqual(weighted_mse_metric.result().numpy(), 0.)

    mae_metric = metric_container.metrics[2]
    self.assertEqual(mae_metric.name, 'out2_mae')
    self.assertEqual(mae_metric.result().numpy(), 2.)

    weighted_mae_metric = metric_container.metrics[3]
    self.assertEqual(weighted_mae_metric.name, 'out2_weighted_mae')
    self.assertEqual(weighted_mae_metric.result().numpy(), 2.)

  def test_metric_partial_dict_with_output_names(self):
    metric_container = compile_utils.MetricsContainer(
        {'out2': 'mae'}, output_names=['out1', 'out2'])

    y_t = [array_ops.ones((10, 1)), array_ops.zeros((10, 1))]
    y_p = [array_ops.ones((10, 1)), array_ops.ones((10, 1))]
    sw = ops.convert_to_tensor_v2([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    metric_container.update_state(y_t, y_p, sample_weight=sw)
    self.assertLen(metric_container.metrics, 1)

    mae_metric = metric_container.metrics[0]
    self.assertEqual(mae_metric.name, 'out2_mae')
    self.assertEqual(mae_metric.result().numpy(), 1.)

  def test_metric_partial_dict_with_nones(self):
    metric_container = compile_utils.MetricsContainer({
        'out1': None,
        'out2': 'mae'
    })

    y_t = {'out1': array_ops.ones((10, 1)), 'out2': array_ops.zeros((10, 1))}
    y_p = {'out1': array_ops.ones((10, 1)), 'out2': array_ops.ones((10, 1))}
    sw = ops.convert_to_tensor_v2([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    metric_container.update_state(y_t, y_p, sample_weight=sw)
    self.assertLen(metric_container.metrics, 1)

    mae_metric = metric_container.metrics[0]
    self.assertEqual(mae_metric.name, 'out2_mae')
    self.assertEqual(mae_metric.result().numpy(), 1.)

  def test_nested_structure(self):
    metric_container = compile_utils.MetricsContainer(
        metrics={
            'b': ['mse', None],
            'a': 'mae'
        },
        weighted_metrics={
            'b': [None, None],
            'a': 'mse'
        })

    y_t = {
        'b': [2 * array_ops.ones((10, 1)),
              array_ops.zeros((10, 1))],
        'a': array_ops.zeros((10, 1))
    }
    y_p = {
        'b': [array_ops.zeros((10, 1)),
              array_ops.zeros((10, 1))],
        'a': array_ops.ones((10, 1))
    }
    sw = ops.convert_to_tensor_v2([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    metric_container.update_state(y_t, y_p, sample_weight=sw)
    self.assertLen(metric_container.metrics, 3)

    a_mae_metric = metric_container.metrics[0]
    self.assertEqual(a_mae_metric.name, 'a_mae')
    self.assertEqual(a_mae_metric.result().numpy(), 1.)

    weighted_a_mae_metric = metric_container.metrics[1]
    self.assertEqual(weighted_a_mae_metric.name, 'a_mse')
    self.assertEqual(weighted_a_mae_metric.result().numpy(), 1.)

    b_1_mse_metric = metric_container.metrics[2]
    self.assertEqual(b_1_mse_metric.name, 'b_1_mse')
    self.assertEqual(b_1_mse_metric.result().numpy(), 4.)

  def test_crossentropy(self):
    metric_container = compile_utils.MetricsContainer('crossentropy')
    y_t, y_p = array_ops.ones((10, 1)), array_ops.ones((10, 1))
    metric_container.update_state(y_t, y_p)
    self.assertEqual(metric_container.metrics[0]._fn,
                     metrics_mod.binary_crossentropy)

    metric_container = compile_utils.MetricsContainer('crossentropy')
    y_t, y_p = array_ops.ones((10, 1)), array_ops.ones((10, 20))
    self.assertEqual(y_p.shape.as_list()[-1], 20)
    metric_container.update_state(y_t, y_p)
    self.assertEqual(metric_container.metrics[0]._fn,
                     metrics_mod.sparse_categorical_crossentropy)

    metric_container = compile_utils.MetricsContainer('crossentropy')
    y_t, y_p = array_ops.ones((10, 20)), array_ops.ones((10, 20))
    metric_container.update_state(y_t, y_p)
    self.assertEqual(metric_container.metrics[0]._fn,
                     metrics_mod.categorical_crossentropy)

  def test_accuracy(self):
    metric_container = compile_utils.MetricsContainer('accuracy')
    y_t, y_p = array_ops.ones((10, 1)), array_ops.ones((10, 1))
    metric_container.update_state(y_t, y_p)
    self.assertEqual(metric_container.metrics[0]._fn,
                     metrics_mod.binary_accuracy)

    metric_container = compile_utils.MetricsContainer('accuracy')
    y_t, y_p = array_ops.ones((10, 1)), array_ops.ones((10, 20))
    self.assertEqual(y_p.shape.as_list()[-1], 20)
    metric_container.update_state(y_t, y_p)
    self.assertEqual(metric_container.metrics[0]._fn,
                     metrics_mod.sparse_categorical_accuracy)

    metric_container = compile_utils.MetricsContainer('accuracy')
    y_t, y_p = array_ops.ones((10, 20)), array_ops.ones((10, 20))
    metric_container.update_state(y_t, y_p)
    self.assertEqual(metric_container.metrics[0]._fn,
                     metrics_mod.categorical_accuracy)

  def test_metric_weighting(self):
    metric_container = compile_utils.MetricsContainer(
        metrics=['mae'], weighted_metrics=['mae'])

    y_t = ops.convert_to_tensor_v2([[0], [3], [0]])
    y_p = ops.convert_to_tensor_v2([[0], [0], [0]])
    sw = ops.convert_to_tensor_v2([[1], [0], [1]])

    metric_container.update_state(y_t, y_p, sample_weight=sw)
    self.assertLen(metric_container.metrics, 2)

    mae_metric = metric_container.metrics[0]
    self.assertEqual(mae_metric.name, 'mae')
    self.assertEqual(mae_metric.result().numpy(), 1.)

    weighted_mae_metric = metric_container.metrics[1]
    self.assertEqual(weighted_mae_metric.name, 'weighted_mae')
    self.assertEqual(weighted_mae_metric.result().numpy(), 0.)

  def test_broadcast_metrics_to_dict(self):
    metric_container = compile_utils.MetricsContainer(metrics=['mae'])

    y_p = {'output': ops.convert_to_tensor([[0], [1], [2]])}
    y_t = {'output': ops.convert_to_tensor([[1], [2], [3]])}
    metric_container.update_state(y_t, y_p)

    mae_metric = metric_container.metrics[0]
    self.assertEqual(mae_metric.name, 'mae')
    self.assertEqual(mae_metric.result().numpy(), 1.)

  def test_broadcast_metrics_to_dict_with_output_names(self):
    metric_container = compile_utils.MetricsContainer(
        metrics=['mae'], output_names=['output'])

    y_p = ops.convert_to_tensor([[0], [1], [2]])
    y_t = {'output': ops.convert_to_tensor([[1], [2], [3]])}
    metric_container.update_state(y_t, y_p)

    mae_metric = metric_container.metrics[0]
    self.assertEqual(mae_metric.name, 'mae')
    self.assertEqual(mae_metric.result().numpy(), 1.)

  def test_missing_label_with_no_metrics(self):
    # It's ok to exclude a label if that label has no
    # losses or metrics associated with it.
    metric_container = compile_utils.MetricsContainer(metrics={
        'output1': 'mae',
        'output3': 'mse'
    })

    y_p = {
        'output1': ops.convert_to_tensor([[0], [1], [2]]),
        'output2': ops.convert_to_tensor([[3], [4], [5]]),
        'output3': ops.convert_to_tensor([[6], [7], [8]])
    }
    y_t = {
        'output1': ops.convert_to_tensor([[1], [2], [3]]),
        'output3': ops.convert_to_tensor([[4], [5], [6]])
    }

    metric_container.update_state(y_t, y_p)
    self.assertLen(metric_container.metrics, 2)

    mae_metric = metric_container.metrics[0]
    self.assertEqual(mae_metric.name, 'output1_mae')
    self.assertEqual(mae_metric.result().numpy(), 1.)

    mse_metric = metric_container.metrics[1]
    self.assertEqual(mse_metric.name, 'output3_mse')
    self.assertEqual(mse_metric.result().numpy(), 4.)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
