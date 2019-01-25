# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras metrics functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import function as eager_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras import metrics
from tensorflow.python.keras import Model
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.checkpointable import util as checkpointable_utils


@test_util.run_all_in_graph_and_eager_modes
class KerasSumTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def test_sum(self):
    m = metrics.Sum(name='my_sum')

    # check config
    self.assertEqual(m.name, 'my_sum')
    self.assertTrue(m.stateful)
    self.assertEqual(m.dtype, dtypes.float32)
    self.assertEqual(len(m.variables), 1)
    self.evaluate(variables.variables_initializer(m.variables))

    # check initial state
    self.assertEqual(self.evaluate(m.total), 0)

    # check __call__()
    self.assertEqual(self.evaluate(m(100)), 100)
    self.assertEqual(self.evaluate(m.total), 100)

    # check update_state() and result() + state accumulation + tensor input
    update_op = m.update_state(ops.convert_n_to_tensor([1, 5]))
    self.evaluate(update_op)
    self.assertAlmostEqual(self.evaluate(m.result()), 106)
    self.assertEqual(self.evaluate(m.total), 106)  # 100 + 1 + 5

    # check reset_states()
    m.reset_states()
    self.assertEqual(self.evaluate(m.total), 0)

  def test_sum_with_sample_weight(self):
    m = metrics.Sum(dtype=dtypes.float64)
    self.assertEqual(m.dtype, dtypes.float64)
    self.evaluate(variables.variables_initializer(m.variables))

    # check scalar weight
    result_t = m(100, sample_weight=0.5)
    self.assertEqual(self.evaluate(result_t), 50)
    self.assertEqual(self.evaluate(m.total), 50)

    # check weights not scalar and weights rank matches values rank
    result_t = m([1, 5], sample_weight=[1, 0.2])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 52., 4)  # 50 + 1 + 5 * 0.2
    self.assertAlmostEqual(self.evaluate(m.total), 52., 4)

    # check weights broadcast
    result_t = m([1, 2], sample_weight=0.5)
    self.assertAlmostEqual(self.evaluate(result_t), 53.5, 1)  # 52 + 0.5 + 1
    self.assertAlmostEqual(self.evaluate(m.total), 53.5, 1)

    # check weights squeeze
    result_t = m([1, 5], sample_weight=[[1], [0.2]])
    self.assertAlmostEqual(self.evaluate(result_t), 55.5, 1)  # 53.5 + 1 + 1
    self.assertAlmostEqual(self.evaluate(m.total), 55.5, 1)

    # check weights expand
    result_t = m([[1], [5]], sample_weight=[1, 0.2])
    self.assertAlmostEqual(self.evaluate(result_t), 57.5, 2)  # 55.5 + 1 + 1
    self.assertAlmostEqual(self.evaluate(m.total), 57.5, 1)

    # check values reduced to the dimensions of weight
    result_t = m([[[1., 2.], [3., 2.], [0.5, 4.]]], sample_weight=[0.5])
    result = np.round(self.evaluate(result_t), decimals=2)
    # result = (prev: 57.5) + 0.5 + 1 + 1.5 + 1 + 0.25 + 2
    self.assertAlmostEqual(result, 63.75, 2)
    self.assertAlmostEqual(self.evaluate(m.total), 63.75, 2)

  def test_sum_graph_with_placeholder(self):
    with context.graph_mode(), self.cached_session() as sess:
      m = metrics.Sum()
      v = array_ops.placeholder(dtypes.float32)
      w = array_ops.placeholder(dtypes.float32)
      self.evaluate(variables.variables_initializer(m.variables))

      # check __call__()
      result_t = m(v, sample_weight=w)
      result = sess.run(result_t, feed_dict=({v: 100, w: 0.5}))
      self.assertEqual(result, 50)
      self.assertEqual(self.evaluate(m.total), 50)

      # check update_state() and result()
      result = sess.run(result_t, feed_dict=({v: [1, 5], w: [1, 0.2]}))
      self.assertAlmostEqual(result, 52., 2)  # 50 + 1 + 5 * 0.2
      self.assertAlmostEqual(self.evaluate(m.total), 52., 2)

  def test_save_restore(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
    m = metrics.Sum()
    checkpoint = checkpointable_utils.Checkpoint(sum=m)
    self.evaluate(variables.variables_initializer(m.variables))

    # update state
    self.evaluate(m(100.))
    self.evaluate(m(200.))

    # save checkpoint and then add an update
    save_path = checkpoint.save(checkpoint_prefix)
    self.evaluate(m(1000.))

    # restore to the same checkpoint sum object (= 300)
    checkpoint.restore(save_path).assert_consumed().run_restore_ops()
    self.evaluate(m(300.))
    self.assertEqual(600., self.evaluate(m.result()))

    # restore to a different checkpoint sum object
    restore_sum = metrics.Sum()
    restore_checkpoint = checkpointable_utils.Checkpoint(sum=restore_sum)
    status = restore_checkpoint.restore(save_path)
    restore_update = restore_sum(300.)
    status.assert_consumed().run_restore_ops()
    self.evaluate(restore_update)
    self.assertEqual(600., self.evaluate(restore_sum.result()))


@test_util.run_all_in_graph_and_eager_modes
class KerasMeanTest(test.TestCase):

  # TODO(b/120949004): Re-enable garbage collection check
  # @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def test_mean(self):
    m = metrics.Mean(name='my_mean')

    # check config
    self.assertEqual(m.name, 'my_mean')
    self.assertTrue(m.stateful)
    self.assertEqual(m.dtype, dtypes.float32)
    self.assertEqual(len(m.variables), 2)
    self.evaluate(variables.variables_initializer(m.variables))

    # check initial state
    self.assertEqual(self.evaluate(m.total), 0)
    self.assertEqual(self.evaluate(m.count), 0)

    # check __call__()
    self.assertEqual(self.evaluate(m(100)), 100)
    self.assertEqual(self.evaluate(m.total), 100)
    self.assertEqual(self.evaluate(m.count), 1)

    # check update_state() and result() + state accumulation + tensor input
    update_op = m.update_state(ops.convert_n_to_tensor([1, 5]))
    self.evaluate(update_op)
    self.assertAlmostEqual(self.evaluate(m.result()), 106 / 3, 2)
    self.assertEqual(self.evaluate(m.total), 106)  # 100 + 1 + 5
    self.assertEqual(self.evaluate(m.count), 3)

    # check reset_states()
    m.reset_states()
    self.assertEqual(self.evaluate(m.total), 0)
    self.assertEqual(self.evaluate(m.count), 0)

    # Check save and restore config
    m2 = metrics.Mean.from_config(m.get_config())
    self.assertEqual(m2.name, 'my_mean')
    self.assertTrue(m2.stateful)
    self.assertEqual(m2.dtype, dtypes.float32)
    self.assertEqual(len(m2.variables), 2)

  def test_mean_with_sample_weight(self):
    m = metrics.Mean(dtype=dtypes.float64)
    self.assertEqual(m.dtype, dtypes.float64)
    self.evaluate(variables.variables_initializer(m.variables))

    # check scalar weight
    result_t = m(100, sample_weight=0.5)
    self.assertEqual(self.evaluate(result_t), 50 / 0.5)
    self.assertEqual(self.evaluate(m.total), 50)
    self.assertEqual(self.evaluate(m.count), 0.5)

    # check weights not scalar and weights rank matches values rank
    result_t = m([1, 5], sample_weight=[1, 0.2])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 52 / 1.7, 2)
    self.assertAlmostEqual(self.evaluate(m.total), 52, 2)  # 50 + 1 + 5 * 0.2
    self.assertAlmostEqual(self.evaluate(m.count), 1.7, 2)  # 0.5 + 1.2

    # check weights broadcast
    result_t = m([1, 2], sample_weight=0.5)
    self.assertAlmostEqual(self.evaluate(result_t), 53.5 / 2.7, 2)
    self.assertAlmostEqual(self.evaluate(m.total), 53.5, 2)  # 52 + 0.5 + 1
    self.assertAlmostEqual(self.evaluate(m.count), 2.7, 2)  # 1.7 + 0.5 + 0.5

    # check weights squeeze
    result_t = m([1, 5], sample_weight=[[1], [0.2]])
    self.assertAlmostEqual(self.evaluate(result_t), 55.5 / 3.9, 2)
    self.assertAlmostEqual(self.evaluate(m.total), 55.5, 2)  # 53.5 + 1 + 1
    self.assertAlmostEqual(self.evaluate(m.count), 3.9, 2)  # 2.7 + 1.2

    # check weights expand
    result_t = m([[1], [5]], sample_weight=[1, 0.2])
    self.assertAlmostEqual(self.evaluate(result_t), 57.5 / 5.1, 2)
    self.assertAlmostEqual(self.evaluate(m.total), 57.5, 2)  # 55.5 + 1 + 1
    self.assertAlmostEqual(self.evaluate(m.count), 5.1, 2)  # 3.9 + 1.2

    # check values reduced to the dimensions of weight
    result_t = m([[[1., 2.], [3., 2.], [0.5, 4.]]], sample_weight=[0.5])
    result = np.round(self.evaluate(result_t), decimals=2)  # 58.5 / 5.6
    self.assertEqual(result, 10.45)
    self.assertEqual(np.round(self.evaluate(m.total), decimals=2), 58.54)
    self.assertEqual(np.round(self.evaluate(m.count), decimals=2), 5.6)

  def test_mean_graph_with_placeholder(self):
    with context.graph_mode(), self.cached_session() as sess:
      m = metrics.Mean()
      v = array_ops.placeholder(dtypes.float32)
      w = array_ops.placeholder(dtypes.float32)
      self.evaluate(variables.variables_initializer(m.variables))

      # check __call__()
      result_t = m(v, sample_weight=w)
      result = sess.run(result_t, feed_dict=({v: 100, w: 0.5}))
      self.assertEqual(self.evaluate(m.total), 50)
      self.assertEqual(self.evaluate(m.count), 0.5)
      self.assertEqual(result, 50 / 0.5)

      # check update_state() and result()
      result = sess.run(result_t, feed_dict=({v: [1, 5], w: [1, 0.2]}))
      self.assertAlmostEqual(self.evaluate(m.total), 52, 2)  # 50 + 1 + 5 * 0.2
      self.assertAlmostEqual(self.evaluate(m.count), 1.7, 2)  # 0.5 + 1.2
      self.assertAlmostEqual(result, 52 / 1.7, 2)

  def test_save_restore(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
    m = metrics.Mean()
    checkpoint = checkpointable_utils.Checkpoint(mean=m)
    self.evaluate(variables.variables_initializer(m.variables))

    # update state
    self.evaluate(m(100.))
    self.evaluate(m(200.))

    # save checkpoint and then add an update
    save_path = checkpoint.save(checkpoint_prefix)
    self.evaluate(m(1000.))

    # restore to the same checkpoint mean object
    checkpoint.restore(save_path).assert_consumed().run_restore_ops()
    self.evaluate(m(300.))
    self.assertEqual(200., self.evaluate(m.result()))

    # restore to a different checkpoint mean object
    restore_mean = metrics.Mean()
    restore_checkpoint = checkpointable_utils.Checkpoint(mean=restore_mean)
    status = restore_checkpoint.restore(save_path)
    restore_update = restore_mean(300.)
    status.assert_consumed().run_restore_ops()
    self.evaluate(restore_update)
    self.assertEqual(200., self.evaluate(restore_mean.result()))
    self.assertEqual(3, self.evaluate(restore_mean.count))


@test_util.run_all_in_graph_and_eager_modes
class KerasAccuracyTest(test.TestCase):

  def test_accuracy(self):
    acc_obj = metrics.Accuracy(name='my acc')

    # check config
    self.assertEqual(acc_obj.name, 'my acc')
    self.assertTrue(acc_obj.stateful)
    self.assertEqual(len(acc_obj.variables), 2)
    self.assertEqual(acc_obj.dtype, dtypes.float32)
    self.evaluate(variables.variables_initializer(acc_obj.variables))

    # verify that correct value is returned
    update_op = acc_obj.update_state([[1], [2], [3], [4]], [[1], [2], [3], [4]])
    self.evaluate(update_op)
    result = self.evaluate(acc_obj.result())
    self.assertEqual(result, 1)  # 2/2

    # Check save and restore config
    a2 = metrics.Accuracy.from_config(acc_obj.get_config())
    self.assertEqual(a2.name, 'my acc')
    self.assertTrue(a2.stateful)
    self.assertEqual(len(a2.variables), 2)
    self.assertEqual(a2.dtype, dtypes.float32)

    # check with sample_weight
    result_t = acc_obj([[2], [1]], [[2], [0]], sample_weight=[[0.5], [0.2]])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.96, 2)  # 4.5/4.7

  def test_binary_accuracy(self):
    acc_obj = metrics.BinaryAccuracy(name='my acc')

    # check config
    self.assertEqual(acc_obj.name, 'my acc')
    self.assertTrue(acc_obj.stateful)
    self.assertEqual(len(acc_obj.variables), 2)
    self.assertEqual(acc_obj.dtype, dtypes.float32)
    self.evaluate(variables.variables_initializer(acc_obj.variables))

    # verify that correct value is returned
    update_op = acc_obj.update_state([[1], [0]], [[1], [0]])
    self.evaluate(update_op)
    result = self.evaluate(acc_obj.result())
    self.assertEqual(result, 1)  # 2/2

    # check y_pred squeeze
    update_op = acc_obj.update_state([[1], [1]], [[[1]], [[0]]])
    self.evaluate(update_op)
    result = self.evaluate(acc_obj.result())
    self.assertAlmostEqual(result, 0.75, 2)  # 3/4

    # check y_true squeeze
    result_t = acc_obj([[[1]], [[1]]], [[1], [0]])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.67, 2)  # 4/6

    # check with sample_weight
    result_t = acc_obj([[1], [1]], [[1], [0]], [[0.5], [0.2]])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.67, 2)  # 4.5/6.7

  def test_binary_accuracy_threshold(self):
    acc_obj = metrics.BinaryAccuracy(threshold=0.7)
    self.evaluate(variables.variables_initializer(acc_obj.variables))
    result_t = acc_obj([[1], [1], [0], [0]], [[0.9], [0.6], [0.4], [0.8]])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.5, 2)

  def test_categorical_accuracy(self):
    acc_obj = metrics.CategoricalAccuracy(name='my acc')

    # check config
    self.assertEqual(acc_obj.name, 'my acc')
    self.assertTrue(acc_obj.stateful)
    self.assertEqual(len(acc_obj.variables), 2)
    self.assertEqual(acc_obj.dtype, dtypes.float32)
    self.evaluate(variables.variables_initializer(acc_obj.variables))

    # verify that correct value is returned
    update_op = acc_obj.update_state([[0, 0, 1], [0, 1, 0]],
                                     [[0.1, 0.1, 0.8], [0.05, 0.95, 0]])
    self.evaluate(update_op)
    result = self.evaluate(acc_obj.result())
    self.assertEqual(result, 1)  # 2/2

    # check with sample_weight
    result_t = acc_obj([[0, 0, 1], [0, 1, 0]],
                       [[0.1, 0.1, 0.8], [0.05, 0, 0.95]], [[0.5], [0.2]])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.93, 2)  # 2.5/2.7

  def test_sparse_categorical_accuracy(self):
    acc_obj = metrics.SparseCategoricalAccuracy(name='my acc')

    # check config
    self.assertEqual(acc_obj.name, 'my acc')
    self.assertTrue(acc_obj.stateful)
    self.assertEqual(len(acc_obj.variables), 2)
    self.assertEqual(acc_obj.dtype, dtypes.float32)
    self.evaluate(variables.variables_initializer(acc_obj.variables))

    # verify that correct value is returned
    update_op = acc_obj.update_state([[2], [1]],
                                     [[0.1, 0.1, 0.8], [0.05, 0.95, 0]])
    self.evaluate(update_op)
    result = self.evaluate(acc_obj.result())
    self.assertEqual(result, 1)  # 2/2

    # check with sample_weight
    result_t = acc_obj([[2], [1]], [[0.1, 0.1, 0.8], [0.05, 0, 0.95]],
                       [[0.5], [0.2]])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.93, 2)  # 2.5/2.7

  def test_sparse_categorical_accuracy_mismatched_dims(self):
    acc_obj = metrics.SparseCategoricalAccuracy(name='my acc')

    # check config
    self.assertEqual(acc_obj.name, 'my acc')
    self.assertTrue(acc_obj.stateful)
    self.assertEqual(len(acc_obj.variables), 2)
    self.assertEqual(acc_obj.dtype, dtypes.float32)
    self.evaluate(variables.variables_initializer(acc_obj.variables))

    # verify that correct value is returned
    update_op = acc_obj.update_state([2, 1], [[0.1, 0.1, 0.8], [0.05, 0.95, 0]])
    self.evaluate(update_op)
    result = self.evaluate(acc_obj.result())
    self.assertEqual(result, 1)  # 2/2

    # check with sample_weight
    result_t = acc_obj([2, 1], [[0.1, 0.1, 0.8], [0.05, 0, 0.95]],
                       [[0.5], [0.2]])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.93, 2)  # 2.5/2.7

  def test_sparse_categorical_accuracy_mismatched_dims_dynamic(self):
    with context.graph_mode(), self.cached_session() as sess:
      acc_obj = metrics.SparseCategoricalAccuracy(name='my acc')
      self.evaluate(variables.variables_initializer(acc_obj.variables))

      t = array_ops.placeholder(dtypes.float32)
      p = array_ops.placeholder(dtypes.float32)
      w = array_ops.placeholder(dtypes.float32)

      result_t = acc_obj(t, p, w)
      result = sess.run(
          result_t,
          feed_dict=({
              t: [2, 1],
              p: [[0.1, 0.1, 0.8], [0.05, 0, 0.95]],
              w: [[0.5], [0.2]]
          }))
      self.assertAlmostEqual(result, 0.71, 2)  # 2.5/2.7


@test_util.run_all_in_graph_and_eager_modes
class CosineProximityTest(test.TestCase):

  def l2_norm(self, x, axis):
    epsilon = 1e-12
    square_sum = np.sum(np.square(x), axis=axis, keepdims=True)
    x_inv_norm = 1 / np.sqrt(np.maximum(square_sum, epsilon))
    return np.multiply(x, x_inv_norm)

  def setup(self, axis=1):
    self.np_y_true = np.asarray([[1, 9, 2], [-5, -2, 6]], dtype=np.float32)
    self.np_y_pred = np.asarray([[4, 8, 12], [8, 1, 3]], dtype=np.float32)

    y_true = self.l2_norm(self.np_y_true, axis)
    y_pred = self.l2_norm(self.np_y_pred, axis)
    self.expected_loss = -np.sum(np.multiply(y_true, y_pred), axis=(axis,))

    self.y_true = constant_op.constant(self.np_y_true)
    self.y_pred = constant_op.constant(self.np_y_pred)

  def test_config(self):
    cosine_obj = metrics.CosineProximity(
        axis=2, name='my_cos', dtype=dtypes.int32)
    self.assertEqual(cosine_obj.name, 'my_cos')
    self.assertEqual(cosine_obj._dtype, dtypes.int32)

    # Check save and restore config
    cosine_obj2 = metrics.CosineProximity.from_config(cosine_obj.get_config())
    self.assertEqual(cosine_obj2.name, 'my_cos')
    self.assertEqual(cosine_obj2._dtype, dtypes.int32)

  def test_unweighted(self):
    self.setup()
    cosine_obj = metrics.CosineProximity()
    self.evaluate(variables.variables_initializer(cosine_obj.variables))
    loss = cosine_obj(self.y_true, self.y_pred)
    expected_loss = np.mean(self.expected_loss)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_weighted(self):
    self.setup()
    cosine_obj = metrics.CosineProximity()
    self.evaluate(variables.variables_initializer(cosine_obj.variables))
    sample_weight = np.asarray([1.2, 3.4])
    loss = cosine_obj(
        self.y_true,
        self.y_pred,
        sample_weight=constant_op.constant(sample_weight))
    expected_loss = np.sum(
        self.expected_loss * sample_weight) / np.sum(sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_axis(self):
    self.setup(axis=1)
    cosine_obj = metrics.CosineProximity(axis=1)
    self.evaluate(variables.variables_initializer(cosine_obj.variables))
    loss = cosine_obj(self.y_true, self.y_pred)
    expected_loss = np.mean(self.expected_loss)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)


@test_util.run_all_in_graph_and_eager_modes
class MeanAbsoluteErrorTest(test.TestCase):

  def test_config(self):
    mae_obj = metrics.MeanAbsoluteError(name='my_mae', dtype=dtypes.int32)
    self.assertEqual(mae_obj.name, 'my_mae')
    self.assertEqual(mae_obj._dtype, dtypes.int32)

    # Check save and restore config
    mae_obj2 = metrics.MeanAbsoluteError.from_config(mae_obj.get_config())
    self.assertEqual(mae_obj2.name, 'my_mae')
    self.assertEqual(mae_obj2._dtype, dtypes.int32)

  def test_unweighted(self):
    mae_obj = metrics.MeanAbsoluteError()
    self.evaluate(variables.variables_initializer(mae_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = mae_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = mae_obj.result()
    self.assertAllClose(0.5, result, atol=1e-5)

  def test_weighted(self):
    mae_obj = metrics.MeanAbsoluteError()
    self.evaluate(variables.variables_initializer(mae_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = constant_op.constant((1., 1.5, 2., 2.5))
    result = mae_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(0.54285, self.evaluate(result), atol=1e-5)


@test_util.run_all_in_graph_and_eager_modes
class MeanAbsolutePercentageErrorTest(test.TestCase):

  def test_config(self):
    mape_obj = metrics.MeanAbsolutePercentageError(
        name='my_mape', dtype=dtypes.int32)
    self.assertEqual(mape_obj.name, 'my_mape')
    self.assertEqual(mape_obj._dtype, dtypes.int32)

    # Check save and restore config
    mape_obj2 = metrics.MeanAbsolutePercentageError.from_config(
        mape_obj.get_config())
    self.assertEqual(mape_obj2.name, 'my_mape')
    self.assertEqual(mape_obj2._dtype, dtypes.int32)

  def test_unweighted(self):
    mape_obj = metrics.MeanAbsolutePercentageError()
    self.evaluate(variables.variables_initializer(mape_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = mape_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = mape_obj.result()
    self.assertAllClose(35e7, result, atol=1e-5)

  def test_weighted(self):
    mape_obj = metrics.MeanAbsolutePercentageError()
    self.evaluate(variables.variables_initializer(mape_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = constant_op.constant((1., 1.5, 2., 2.5))
    result = mape_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(40e7, self.evaluate(result), atol=1e-5)


@test_util.run_all_in_graph_and_eager_modes
class MeanSquaredErrorTest(test.TestCase):

  def test_config(self):
    mse_obj = metrics.MeanSquaredError(name='my_mse', dtype=dtypes.int32)
    self.assertEqual(mse_obj.name, 'my_mse')
    self.assertEqual(mse_obj._dtype, dtypes.int32)

    # Check save and restore config
    mse_obj2 = metrics.MeanSquaredError.from_config(mse_obj.get_config())
    self.assertEqual(mse_obj2.name, 'my_mse')
    self.assertEqual(mse_obj2._dtype, dtypes.int32)

  def test_unweighted(self):
    mse_obj = metrics.MeanSquaredError()
    self.evaluate(variables.variables_initializer(mse_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = mse_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = mse_obj.result()
    self.assertAllClose(0.5, result, atol=1e-5)

  def test_weighted(self):
    mse_obj = metrics.MeanSquaredError()
    self.evaluate(variables.variables_initializer(mse_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = constant_op.constant((1., 1.5, 2., 2.5))
    result = mse_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(0.54285, self.evaluate(result), atol=1e-5)


@test_util.run_all_in_graph_and_eager_modes
class MeanSquaredLogarithmicErrorTest(test.TestCase):

  def test_config(self):
    msle_obj = metrics.MeanSquaredLogarithmicError(
        name='my_msle', dtype=dtypes.int32)
    self.assertEqual(msle_obj.name, 'my_msle')
    self.assertEqual(msle_obj._dtype, dtypes.int32)

    # Check save and restore config
    msle_obj2 = metrics.MeanSquaredLogarithmicError.from_config(
        msle_obj.get_config())
    self.assertEqual(msle_obj2.name, 'my_msle')
    self.assertEqual(msle_obj2._dtype, dtypes.int32)

  def test_unweighted(self):
    msle_obj = metrics.MeanSquaredLogarithmicError()
    self.evaluate(variables.variables_initializer(msle_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = msle_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = msle_obj.result()
    self.assertAllClose(0.24022, result, atol=1e-5)

  def test_weighted(self):
    msle_obj = metrics.MeanSquaredLogarithmicError()
    self.evaluate(variables.variables_initializer(msle_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = constant_op.constant((1., 1.5, 2., 2.5))
    result = msle_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(0.26082, self.evaluate(result), atol=1e-5)


@test_util.run_all_in_graph_and_eager_modes
class HingeTest(test.TestCase):

  def test_config(self):
    hinge_obj = metrics.Hinge(name='hinge', dtype=dtypes.int32)
    self.assertEqual(hinge_obj.name, 'hinge')
    self.assertEqual(hinge_obj._dtype, dtypes.int32)

    # Check save and restore config
    hinge_obj2 = metrics.Hinge.from_config(hinge_obj.get_config())
    self.assertEqual(hinge_obj2.name, 'hinge')
    self.assertEqual(hinge_obj2._dtype, dtypes.int32)

  def test_unweighted(self):
    hinge_obj = metrics.Hinge()
    self.evaluate(variables.variables_initializer(hinge_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = hinge_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = hinge_obj.result()
    self.assertAllClose(0.65, result, atol=1e-5)

  def test_weighted(self):
    hinge_obj = metrics.Hinge()
    self.evaluate(variables.variables_initializer(hinge_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = constant_op.constant((1., 1.5, 2., 2.5))
    result = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(0.65714, self.evaluate(result), atol=1e-5)


@test_util.run_all_in_graph_and_eager_modes
class SquaredHingeTest(test.TestCase):

  def test_config(self):
    sq_hinge_obj = metrics.SquaredHinge(name='sq_hinge', dtype=dtypes.int32)
    self.assertEqual(sq_hinge_obj.name, 'sq_hinge')
    self.assertEqual(sq_hinge_obj._dtype, dtypes.int32)

    # Check save and restore config
    sq_hinge_obj2 = metrics.SquaredHinge.from_config(sq_hinge_obj.get_config())
    self.assertEqual(sq_hinge_obj2.name, 'sq_hinge')
    self.assertEqual(sq_hinge_obj2._dtype, dtypes.int32)

  def test_unweighted(self):
    sq_hinge_obj = metrics.SquaredHinge()
    self.evaluate(variables.variables_initializer(sq_hinge_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = sq_hinge_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = sq_hinge_obj.result()
    self.assertAllClose(0.65, result, atol=1e-5)

  def test_weighted(self):
    sq_hinge_obj = metrics.SquaredHinge()
    self.evaluate(variables.variables_initializer(sq_hinge_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = constant_op.constant((1., 1.5, 2., 2.5))
    result = sq_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(0.65714, self.evaluate(result), atol=1e-5)


@test_util.run_all_in_graph_and_eager_modes
class CategoricalHingeTest(test.TestCase):

  def test_config(self):
    cat_hinge_obj = metrics.CategoricalHinge(
        name='cat_hinge', dtype=dtypes.int32)
    self.assertEqual(cat_hinge_obj.name, 'cat_hinge')
    self.assertEqual(cat_hinge_obj._dtype, dtypes.int32)

    # Check save and restore config
    cat_hinge_obj2 = metrics.CategoricalHinge.from_config(
        cat_hinge_obj.get_config())
    self.assertEqual(cat_hinge_obj2.name, 'cat_hinge')
    self.assertEqual(cat_hinge_obj2._dtype, dtypes.int32)

  def test_unweighted(self):
    cat_hinge_obj = metrics.CategoricalHinge()
    self.evaluate(variables.variables_initializer(cat_hinge_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = cat_hinge_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = cat_hinge_obj.result()
    self.assertAllClose(0.5, result, atol=1e-5)

  def test_weighted(self):
    cat_hinge_obj = metrics.CategoricalHinge()
    self.evaluate(variables.variables_initializer(cat_hinge_obj.variables))
    y_true = constant_op.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = constant_op.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = constant_op.constant((1., 1.5, 2., 2.5))
    result = cat_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(0.5, self.evaluate(result), atol=1e-5)


@test_util.run_all_in_graph_and_eager_modes
class RootMeanSquaredErrorTest(test.TestCase):

  def test_config(self):
    rmse_obj = metrics.RootMeanSquaredError(name='rmse', dtype=dtypes.int32)
    self.assertEqual(rmse_obj.name, 'rmse')
    self.assertEqual(rmse_obj._dtype, dtypes.int32)

    rmse_obj2 = metrics.RootMeanSquaredError.from_config(rmse_obj.get_config())
    self.assertEqual(rmse_obj2.name, 'rmse')
    self.assertEqual(rmse_obj2._dtype, dtypes.int32)

  def test_unweighted(self):
    rmse_obj = metrics.RootMeanSquaredError()
    self.evaluate(variables.variables_initializer(rmse_obj.variables))
    y_true = constant_op.constant((2, 4, 6))
    y_pred = constant_op.constant((1, 3, 2))

    update_op = rmse_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = rmse_obj.result()
    # error = [-1, -1, -4], square(error) = [1, 1, 16], mean = 18/3 = 6
    self.assertAllClose(math.sqrt(6), result, atol=1e-3)

  def test_weighted(self):
    rmse_obj = metrics.RootMeanSquaredError()
    self.evaluate(variables.variables_initializer(rmse_obj.variables))
    y_true = constant_op.constant((2, 4, 6, 8))
    y_pred = constant_op.constant((1, 3, 2, 3))
    sample_weight = constant_op.constant((0, 1, 0, 1))
    result = rmse_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(math.sqrt(13), self.evaluate(result), atol=1e-3)


@test_util.run_all_in_graph_and_eager_modes
class TopKCategoricalAccuracyTest(test.TestCase):

  def test_config(self):
    a_obj = metrics.TopKCategoricalAccuracy(name='topkca', dtype=dtypes.int32)
    self.assertEqual(a_obj.name, 'topkca')
    self.assertEqual(a_obj._dtype, dtypes.int32)

    a_obj2 = metrics.TopKCategoricalAccuracy.from_config(a_obj.get_config())
    self.assertEqual(a_obj2.name, 'topkca')
    self.assertEqual(a_obj2._dtype, dtypes.int32)

  def test_correctness(self):
    a_obj = metrics.TopKCategoricalAccuracy()
    self.evaluate(variables.variables_initializer(a_obj.variables))
    y_true = constant_op.constant([[0, 0, 1], [0, 1, 0]])
    y_pred = constant_op.constant([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])

    result = a_obj(y_true, y_pred)
    self.assertEqual(1, self.evaluate(result))  # both the samples match

    # With `k` < 5.
    a_obj = metrics.TopKCategoricalAccuracy(k=1)
    self.evaluate(variables.variables_initializer(a_obj.variables))
    result = a_obj(y_true, y_pred)
    self.assertEqual(0.5, self.evaluate(result))  # only sample #2 matches

    # With `k` > 5.
    y_true = constant_op.constant([[0, 0, 1, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0, 0]])
    y_pred = constant_op.constant([[0.5, 0.9, 0.1, 0.7, 0.6, 0.5, 0.4],
                                   [0.05, 0.95, 0, 0, 0, 0, 0]])
    a_obj = metrics.TopKCategoricalAccuracy(k=6)
    self.evaluate(variables.variables_initializer(a_obj.variables))
    result = a_obj(y_true, y_pred)
    self.assertEqual(0.5, self.evaluate(result))  # only 1 sample matches.


@test_util.run_all_in_graph_and_eager_modes
class SparseTopKCategoricalAccuracyTest(test.TestCase):

  def test_config(self):
    a_obj = metrics.SparseTopKCategoricalAccuracy(
        name='stopkca', dtype=dtypes.int32)
    self.assertEqual(a_obj.name, 'stopkca')
    self.assertEqual(a_obj._dtype, dtypes.int32)

    a_obj2 = metrics.SparseTopKCategoricalAccuracy.from_config(
        a_obj.get_config())
    self.assertEqual(a_obj2.name, 'stopkca')
    self.assertEqual(a_obj2._dtype, dtypes.int32)

  def test_correctness(self):
    a_obj = metrics.SparseTopKCategoricalAccuracy()
    self.evaluate(variables.variables_initializer(a_obj.variables))
    y_true = constant_op.constant([2, 1])
    y_pred = constant_op.constant([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])

    result = a_obj(y_true, y_pred)
    self.assertEqual(1, self.evaluate(result))  # both the samples match

    # With `k` < 5.
    a_obj = metrics.SparseTopKCategoricalAccuracy(k=1)
    self.evaluate(variables.variables_initializer(a_obj.variables))
    result = a_obj(y_true, y_pred)
    self.assertEqual(0.5, self.evaluate(result))  # only sample #2 matches

    # With `k` > 5.
    y_pred = constant_op.constant([[0.5, 0.9, 0.1, 0.7, 0.6, 0.5, 0.4],
                                   [0.05, 0.95, 0, 0, 0, 0, 0]])
    a_obj = metrics.SparseTopKCategoricalAccuracy(k=6)
    self.evaluate(variables.variables_initializer(a_obj.variables))
    result = a_obj(y_true, y_pred)
    self.assertEqual(0.5, self.evaluate(result))  # only 1 sample matches.


@test_util.run_all_in_graph_and_eager_modes
class LogcoshTest(test.TestCase):

  def setup(self):
    y_pred = np.asarray([1, 9, 2, -5, -2, 6]).reshape((2, 3))
    y_true = np.asarray([4, 8, 12, 8, 1, 3]).reshape((2, 3))

    self.batch_size = 6
    error = y_pred - y_true
    self.expected_results = np.log((np.exp(error) + np.exp(-error)) / 2)

    self.y_pred = constant_op.constant(y_pred, dtype=dtypes.float32)
    self.y_true = constant_op.constant(y_true)

  def test_config(self):
    logcosh_obj = metrics.Logcosh(name='logcosh', dtype=dtypes.int32)
    self.assertEqual(logcosh_obj.name, 'logcosh')
    self.assertEqual(logcosh_obj._dtype, dtypes.int32)

  def test_unweighted(self):
    self.setup()
    logcosh_obj = metrics.Logcosh()
    self.evaluate(variables.variables_initializer(logcosh_obj.variables))

    update_op = logcosh_obj.update_state(self.y_true, self.y_pred)
    self.evaluate(update_op)
    result = logcosh_obj.result()
    expected_result = np.sum(self.expected_results) / self.batch_size
    self.assertAllClose(result, expected_result, atol=1e-3)

  def test_weighted(self):
    self.setup()
    logcosh_obj = metrics.Logcosh()
    self.evaluate(variables.variables_initializer(logcosh_obj.variables))
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    result = logcosh_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

    sample_weight = np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3))
    expected_result = np.multiply(self.expected_results, sample_weight)
    expected_result = np.sum(expected_result) / np.sum(sample_weight)
    self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


@test_util.run_all_in_graph_and_eager_modes
class PoissonTest(test.TestCase):

  def setup(self):
    y_pred = np.asarray([1, 9, 2, 5, 2, 6]).reshape((2, 3))
    y_true = np.asarray([4, 8, 12, 8, 1, 3]).reshape((2, 3))

    self.batch_size = 6
    self.expected_results = y_pred - np.multiply(y_true, np.log(y_pred))

    self.y_pred = constant_op.constant(y_pred, dtype=dtypes.float32)
    self.y_true = constant_op.constant(y_true)

  def test_config(self):
    poisson_obj = metrics.Poisson(name='poisson', dtype=dtypes.int32)
    self.assertEqual(poisson_obj.name, 'poisson')
    self.assertEqual(poisson_obj._dtype, dtypes.int32)

    poisson_obj2 = metrics.Poisson.from_config(poisson_obj.get_config())
    self.assertEqual(poisson_obj2.name, 'poisson')
    self.assertEqual(poisson_obj2._dtype, dtypes.int32)

  def test_unweighted(self):
    self.setup()
    poisson_obj = metrics.Poisson()
    self.evaluate(variables.variables_initializer(poisson_obj.variables))

    update_op = poisson_obj.update_state(self.y_true, self.y_pred)
    self.evaluate(update_op)
    result = poisson_obj.result()
    expected_result = np.sum(self.expected_results) / self.batch_size
    self.assertAllClose(result, expected_result, atol=1e-3)

  def test_weighted(self):
    self.setup()
    poisson_obj = metrics.Poisson()
    self.evaluate(variables.variables_initializer(poisson_obj.variables))
    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))

    result = poisson_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    sample_weight = np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3))
    expected_result = np.multiply(self.expected_results, sample_weight)
    expected_result = np.sum(expected_result) / np.sum(sample_weight)
    self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


@test_util.run_all_in_graph_and_eager_modes
class KullbackLeiblerDivergenceTest(test.TestCase):

  def setup(self):
    y_pred = np.asarray([.4, .9, .12, .36, .3, .4]).reshape((2, 3))
    y_true = np.asarray([.5, .8, .12, .7, .43, .8]).reshape((2, 3))

    self.batch_size = 2
    self.expected_results = np.multiply(y_true, np.log(y_true / y_pred))

    self.y_pred = constant_op.constant(y_pred, dtype=dtypes.float32)
    self.y_true = constant_op.constant(y_true)

  def test_config(self):
    k_obj = metrics.KullbackLeiblerDivergence(name='kld', dtype=dtypes.int32)
    self.assertEqual(k_obj.name, 'kld')
    self.assertEqual(k_obj._dtype, dtypes.int32)

    k_obj2 = metrics.KullbackLeiblerDivergence.from_config(k_obj.get_config())
    self.assertEqual(k_obj2.name, 'kld')
    self.assertEqual(k_obj2._dtype, dtypes.int32)

  def test_unweighted(self):
    self.setup()
    k_obj = metrics.KullbackLeiblerDivergence()
    self.evaluate(variables.variables_initializer(k_obj.variables))

    update_op = k_obj.update_state(self.y_true, self.y_pred)
    self.evaluate(update_op)
    result = k_obj.result()
    expected_result = np.sum(self.expected_results) / self.batch_size
    self.assertAllClose(result, expected_result, atol=1e-3)

  def test_weighted(self):
    self.setup()
    k_obj = metrics.KullbackLeiblerDivergence()
    self.evaluate(variables.variables_initializer(k_obj.variables))

    sample_weight = constant_op.constant([1.2, 3.4], shape=(2, 1))
    result = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

    sample_weight = np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3))
    expected_result = np.multiply(self.expected_results, sample_weight)
    expected_result = np.sum(expected_result) / (1.2 + 3.4)
    self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


@test_util.run_all_in_graph_and_eager_modes
class MeanRelativeErrorTest(test.TestCase):

  def test_config(self):
    normalizer = constant_op.constant([1, 3], dtype=dtypes.float32)
    mre_obj = metrics.MeanRelativeError(normalizer=normalizer, name='mre')
    self.assertEqual(mre_obj.name, 'mre')
    self.assertArrayNear(self.evaluate(mre_obj.normalizer), [1, 3], 1e-1)

    mre_obj2 = metrics.MeanRelativeError.from_config(mre_obj.get_config())
    self.assertEqual(mre_obj2.name, 'mre')
    self.assertArrayNear(self.evaluate(mre_obj2.normalizer), [1, 3], 1e-1)

  def test_unweighted(self):
    np_y_pred = np.asarray([2, 4, 6, 8], dtype=np.float32)
    np_y_true = np.asarray([1, 3, 2, 3], dtype=np.float32)
    expected_error = np.mean(
        np.divide(np.absolute(np_y_pred - np_y_true), np_y_true))

    y_pred = constant_op.constant(np_y_pred, shape=(1, 4), dtype=dtypes.float32)
    y_true = constant_op.constant(np_y_true, shape=(1, 4))

    mre_obj = metrics.MeanRelativeError(normalizer=y_true)
    self.evaluate(variables.variables_initializer(mre_obj.variables))

    result = mre_obj(y_true, y_pred)
    self.assertAllClose(self.evaluate(result), expected_error, atol=1e-3)

  def test_weighted(self):
    np_y_pred = np.asarray([2, 4, 6, 8], dtype=np.float32)
    np_y_true = np.asarray([1, 3, 2, 3], dtype=np.float32)
    sample_weight = np.asarray([0.2, 0.3, 0.5, 0], dtype=np.float32)
    rel_errors = np.divide(np.absolute(np_y_pred - np_y_true), np_y_true)
    expected_error = np.sum(rel_errors * sample_weight)

    y_pred = constant_op.constant(np_y_pred, dtype=dtypes.float32)
    y_true = constant_op.constant(np_y_true)

    mre_obj = metrics.MeanRelativeError(normalizer=y_true)
    self.evaluate(variables.variables_initializer(mre_obj.variables))

    result = mre_obj(
        y_true, y_pred, sample_weight=constant_op.constant(sample_weight))
    self.assertAllClose(self.evaluate(result), expected_error, atol=1e-3)

  def test_zero_normalizer(self):
    y_pred = constant_op.constant([2, 4], dtype=dtypes.float32)
    y_true = constant_op.constant([1, 3])

    mre_obj = metrics.MeanRelativeError(normalizer=array_ops.zeros_like(y_true))
    self.evaluate(variables.variables_initializer(mre_obj.variables))

    result = mre_obj(y_true, y_pred)
    self.assertEqual(self.evaluate(result), 0)


@test_util.run_all_in_graph_and_eager_modes
class MeanIoUTest(test.TestCase):

  def test_config(self):
    m_obj = metrics.MeanIoU(num_classes=2, name='mean_iou')
    self.assertEqual(m_obj.name, 'mean_iou')
    self.assertEqual(m_obj.num_classes, 2)

    m_obj2 = metrics.MeanIoU.from_config(m_obj.get_config())
    self.assertEqual(m_obj2.name, 'mean_iou')
    self.assertEqual(m_obj2.num_classes, 2)

  def test_unweighted(self):
    y_pred = constant_op.constant([0, 1, 0, 1], dtype=dtypes.float32)
    y_true = constant_op.constant([0, 0, 1, 1])

    m_obj = metrics.MeanIoU(num_classes=2)
    self.evaluate(variables.variables_initializer(m_obj.variables))

    result = m_obj(y_true, y_pred)

    # cm = [[1, 1],
    #       [1, 1]]
    # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
    # iou = true_positives / (sum_row + sum_col - true_positives))
    expected_result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2
    self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

  def test_weighted(self):
    y_pred = constant_op.constant([0, 1, 0, 1], dtype=dtypes.float32)
    y_true = constant_op.constant([0, 0, 1, 1])
    sample_weight = constant_op.constant([0.2, 0.3, 0.4, 0.1])

    m_obj = metrics.MeanIoU(num_classes=2)
    self.evaluate(variables.variables_initializer(m_obj.variables))

    result = m_obj(y_true, y_pred, sample_weight=sample_weight)

    # cm = [[0.2, 0.3],
    #       [0.4, 0.1]]
    # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2, 0.1]
    # iou = true_positives / (sum_row + sum_col - true_positives))
    expected_result = (0.2 / (0.6 + 0.5 - 0.2) + 0.1 / (0.4 + 0.5 - 0.1)) / 2
    self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

  def test_multi_dim_input(self):
    y_pred = constant_op.constant([[0, 1], [0, 1]], dtype=dtypes.float32)
    y_true = constant_op.constant([[0, 0], [1, 1]])
    sample_weight = constant_op.constant([[0.2, 0.3], [0.4, 0.1]])

    m_obj = metrics.MeanIoU(num_classes=2)
    self.evaluate(variables.variables_initializer(m_obj.variables))

    result = m_obj(y_true, y_pred, sample_weight=sample_weight)

    # cm = [[0.2, 0.3],
    #       [0.4, 0.1]]
    # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2, 0.1]
    # iou = true_positives / (sum_row + sum_col - true_positives))
    expected_result = (0.2 / (0.6 + 0.5 - 0.2) + 0.1 / (0.4 + 0.5 - 0.1)) / 2
    self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

  def test_zero_valid_entries(self):
    m_obj = metrics.MeanIoU(num_classes=2)
    self.evaluate(variables.variables_initializer(m_obj.variables))
    self.assertAllClose(self.evaluate(m_obj.result()), 0, atol=1e-3)

  def test_zero_and_non_zero_entries(self):
    y_pred = constant_op.constant([1], dtype=dtypes.float32)
    y_true = constant_op.constant([1])

    m_obj = metrics.MeanIoU(num_classes=2)
    self.evaluate(variables.variables_initializer(m_obj.variables))
    result = m_obj(y_true, y_pred)

    # cm = [[0, 0],
    #       [0, 1]]
    # sum_row = [0, 1], sum_col = [0, 1], true_positives = [0, 1]
    # iou = true_positives / (sum_row + sum_col - true_positives))
    expected_result = (0 + 1 / (1 + 1 - 1)) / 1
    self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


class MeanTensorTest(keras_parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_config(self):
    m = metrics.MeanTensor(name='mean_by_element')

    # check config
    self.assertEqual(m.name, 'mean_by_element')
    self.assertTrue(m.stateful)
    self.assertEqual(m.dtype, dtypes.float32)
    self.assertEqual(len(m.variables), 0)

    with self.assertRaisesRegexp(ValueError, 'does not have any result yet'):
      m.result()

    self.evaluate(m([[3], [5], [3]]))
    self.assertAllEqual(m._shape, [3, 1])

    m2 = metrics.MeanTensor.from_config(m.get_config())
    self.assertEqual(m2.name, 'mean_by_element')
    self.assertTrue(m2.stateful)
    self.assertEqual(m2.dtype, dtypes.float32)
    self.assertEqual(len(m2.variables), 0)

  @test_util.run_in_graph_and_eager_modes
  def test_unweighted(self):
    m = metrics.MeanTensor(dtype=dtypes.float64)

    # check __call__()
    self.assertAllClose(self.evaluate(m([100, 40])), [100, 40])
    self.assertAllClose(self.evaluate(m.total), [100, 40])
    self.assertAllClose(self.evaluate(m.count), [1, 1])

    # check update_state() and result() + state accumulation + tensor input
    update_op = m.update_state(ops.convert_n_to_tensor([1, 5]))
    self.evaluate(update_op)
    self.assertAllClose(self.evaluate(m.result()), [50.5, 22.5])
    self.assertAllClose(self.evaluate(m.total), [101, 45])
    self.assertAllClose(self.evaluate(m.count), [2, 2])

    # check reset_states()
    m.reset_states()
    self.assertAllClose(self.evaluate(m.total), [0, 0])
    self.assertAllClose(self.evaluate(m.count), [0, 0])

  @test_util.run_in_graph_and_eager_modes
  def test_weighted(self):
    m = metrics.MeanTensor(dtype=dtypes.float64)
    self.assertEqual(m.dtype, dtypes.float64)

    # check scalar weight
    result_t = m([100, 30], sample_weight=0.5)
    self.assertAllClose(self.evaluate(result_t), [100, 30])
    self.assertAllClose(self.evaluate(m.total), [50, 15])
    self.assertAllClose(self.evaluate(m.count), [0.5, 0.5])

    # check weights not scalar and weights rank matches values rank
    result_t = m([1, 5], sample_weight=[1, 0.2])
    result = self.evaluate(result_t)
    self.assertAllClose(result, [51 / 1.5, 16 / 0.7], 2)
    self.assertAllClose(self.evaluate(m.total), [51, 16])
    self.assertAllClose(self.evaluate(m.count), [1.5, 0.7])

    # check weights broadcast
    result_t = m([1, 2], sample_weight=0.5)
    self.assertAllClose(self.evaluate(result_t), [51.5 / 2, 17 / 1.2])
    self.assertAllClose(self.evaluate(m.total), [51.5, 17])
    self.assertAllClose(self.evaluate(m.count), [2, 1.2])

    # check weights squeeze
    result_t = m([1, 5], sample_weight=[[1], [0.2]])
    self.assertAllClose(self.evaluate(result_t), [52.5 / 3, 18 / 1.4])
    self.assertAllClose(self.evaluate(m.total), [52.5, 18])
    self.assertAllClose(self.evaluate(m.count), [3, 1.4])

    # check weights expand
    m = metrics.MeanTensor((2, 1), dtype=dtypes.float64)
    self.evaluate(variables.variables_initializer(m.variables))
    result_t = m([[1], [5]], sample_weight=[1, 0.2])
    self.assertAllClose(self.evaluate(result_t), [[1], [5]])
    self.assertAllClose(self.evaluate(m.total), [[1], [1]])
    self.assertAllClose(self.evaluate(m.count), [[1], [0.2]])

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_value_shape(self):
    m = metrics.MeanTensor(dtype=dtypes.float64)
    m([1])
    with self.assertRaisesRegexp(
        ValueError, 'MeanTensor input values must always have the same shape'):
      m([1, 5])

  @test_util.run_in_graph_and_eager_modes
  def test_build_in_tf_function(self):
    """Ensure that variables are created correctly in a tf function."""
    m = metrics.MeanTensor(dtype=dtypes.float64)

    @eager_function.defun
    def call_metric(x):
      return m(x)

    self.assertAllClose(self.evaluate(call_metric([100, 40])), [100, 40])
    self.assertAllClose(self.evaluate(m.total), [100, 40])
    self.assertAllClose(self.evaluate(m.count), [1, 1])
    self.assertAllClose(self.evaluate(call_metric([20, 2])), [60, 21])

  def test_in_keras_model(self):
    with context.eager_mode():
      class ModelWithMetric(Model):

        def __init__(self):
          super(ModelWithMetric, self).__init__()
          self.dense1 = layers.Dense(
              3, activation='relu', kernel_initializer='ones')
          self.dense2 = layers.Dense(
              1, activation='sigmoid', kernel_initializer='ones')
          self.mean_tensor = metrics.MeanTensor()

        def call(self, x):
          x = self.dense1(x)
          x = self.dense2(x)
          self.mean_tensor(self.dense1.kernel)
          return x

      model = ModelWithMetric()
      model.compile(
          loss='mae',
          optimizer='rmsprop',
          run_eagerly=True)

      x = np.ones((100, 4))
      y = np.zeros((100, 1))
      model.evaluate(x, y, batch_size=50)
      self.assertAllClose(self.evaluate(model.mean_tensor.result()),
                          np.ones((4, 3)))
      self.assertAllClose(self.evaluate(model.mean_tensor.total),
                          np.full((4, 3), 2))
      self.assertAllClose(self.evaluate(model.mean_tensor.count),
                          np.full((4, 3), 2))

      model.evaluate(x, y, batch_size=25)
      self.assertAllClose(self.evaluate(model.mean_tensor.result()),
                          np.ones((4, 3)))
      self.assertAllClose(self.evaluate(model.mean_tensor.total),
                          np.full((4, 3), 4))
      self.assertAllClose(self.evaluate(model.mean_tensor.count),
                          np.full((4, 3), 4))


@test_util.run_all_in_graph_and_eager_modes
class BinaryCrossentropyTest(test.TestCase):

  def test_config(self):
    bce_obj = metrics.BinaryCrossentropy(
        name='bce', dtype=dtypes.int32, label_smoothing=0.2)
    self.assertEqual(bce_obj.name, 'bce')
    self.assertEqual(bce_obj._dtype, dtypes.int32)

    old_config = bce_obj.get_config()
    self.assertAllClose(old_config['label_smoothing'], 0.2, 1e-3)

    # Check save and restore config
    bce_obj2 = metrics.BinaryCrossentropy.from_config(old_config)
    self.assertEqual(bce_obj2.name, 'bce')
    self.assertEqual(bce_obj2._dtype, dtypes.int32)
    new_config = bce_obj2.get_config()
    self.assertDictEqual(old_config, new_config)

  def test_unweighted(self):
    bce_obj = metrics.BinaryCrossentropy()
    self.evaluate(variables.variables_initializer(bce_obj.variables))
    y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
    y_pred = np.asarray([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
    result = bce_obj(y_true, y_pred)

    # EPSILON = 1e-7, y = y_true, y` = y_pred, Y_MAX = 0.9999999
    # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    # y` = [Y_MAX, Y_MAX, Y_MAX, EPSILON]

    # Metric = -(y log(y` + EPSILON) + (1 - y) log(1 - y` + EPSILON))
    #        = [-log(Y_MAX + EPSILON), -log(1 - Y_MAX + EPSILON),
    #           -log(Y_MAX + EPSILON), -log(1)]
    #        = [(0 + 15.33) / 2, (0 + 0) / 2]
    # Reduced metric = 7.665 / 2

    self.assertAllClose(self.evaluate(result), 3.833, atol=1e-3)

  def test_unweighted_with_logits(self):
    bce_obj = metrics.BinaryCrossentropy(from_logits=True)
    self.evaluate(variables.variables_initializer(bce_obj.variables))

    y_true = constant_op.constant([[1, 0, 1], [0, 1, 1]])
    y_pred = constant_op.constant([[100.0, -100.0, 100.0],
                                   [100.0, 100.0, -100.0]])
    result = bce_obj(y_true, y_pred)

    # Metric = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    #              (where x = logits and z = y_true)
    #        = [((100 - 100 * 1 + log(1 + exp(-100))) +
    #            (0 + 100 * 0 + log(1 + exp(-100))) +
    #            (100 - 100 * 1 + log(1 + exp(-100))),
    #           ((100 - 100 * 0 + log(1 + exp(-100))) +
    #            (100 - 100 * 1 + log(1 + exp(-100))) +
    #            (0 + 100 * 1 + log(1 + exp(-100))))]
    #        = [(0 + 0 + 0) / 3, 200 / 3]
    # Reduced metric = (0 + 66.666) / 2

    self.assertAllClose(self.evaluate(result), 33.333, atol=1e-3)

  def test_weighted(self):
    bce_obj = metrics.BinaryCrossentropy()
    self.evaluate(variables.variables_initializer(bce_obj.variables))
    y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
    y_pred = np.asarray([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
    sample_weight = constant_op.constant([1.5, 2.])
    result = bce_obj(y_true, y_pred, sample_weight=sample_weight)

    # EPSILON = 1e-7, y = y_true, y` = y_pred, Y_MAX = 0.9999999
    # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    # y` = [Y_MAX, Y_MAX, Y_MAX, EPSILON]

    # Metric = -(y log(y` + EPSILON) + (1 - y) log(1 - y` + EPSILON))
    #        = [-log(Y_MAX + EPSILON), -log(1 - Y_MAX + EPSILON),
    #           -log(Y_MAX + EPSILON), -log(1)]
    #        = [(0 + 15.33) / 2, (0 + 0) / 2]
    # Weighted metric = [7.665 * 1.5, 0]
    # Reduced metric = 7.665 * 1.5 / (1.5 + 2)

    self.assertAllClose(self.evaluate(result), 3.285, atol=1e-3)

  def test_weighted_from_logits(self):
    bce_obj = metrics.BinaryCrossentropy(from_logits=True)
    self.evaluate(variables.variables_initializer(bce_obj.variables))
    y_true = constant_op.constant([[1, 0, 1], [0, 1, 1]])
    y_pred = constant_op.constant([[100.0, -100.0, 100.0],
                                   [100.0, 100.0, -100.0]])
    sample_weight = constant_op.constant([2., 2.5])
    result = bce_obj(y_true, y_pred, sample_weight=sample_weight)

    # Metric = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    #              (where x = logits and z = y_true)
    #        = [(0 + 0 + 0) / 3, 200 / 3]
    # Weighted metric = [0, 66.666 * 2.5]
    # Reduced metric = 66.666 * 2.5 / (2 + 2.5)

    self.assertAllClose(self.evaluate(result), 37.037, atol=1e-3)

  def test_label_smoothing(self):
    logits = constant_op.constant(((100., -100., -100.)))
    y_true = constant_op.constant(((1, 0, 1)))
    label_smoothing = 0.1
    # Metric: max(x, 0) - x * z + log(1 + exp(-abs(x)))
    #             (where x = logits and z = y_true)
    # Label smoothing: z' = z * (1 - L) + 0.5L
    # After label smoothing, label 1 becomes 1 - 0.5L
    #                        label 0 becomes 0.5L
    # Applying the above two fns to the given input:
    # (100 - 100 * (1 - 0.5 L)  + 0 +
    #  0   + 100 * (0.5 L)      + 0 +
    #  0   + 100 * (1 - 0.5 L)  + 0) * (1/3)
    #  = (100 + 50L) * 1/3
    bce_obj = metrics.BinaryCrossentropy(
        from_logits=True, label_smoothing=label_smoothing)
    self.evaluate(variables.variables_initializer(bce_obj.variables))
    result = bce_obj(y_true, logits)
    expected_value = (100.0 + 50.0 * label_smoothing) / 3.0
    self.assertAllClose(expected_value, self.evaluate(result), atol=1e-3)


@test_util.run_all_in_graph_and_eager_modes
class CategoricalCrossentropyTest(test.TestCase):

  def test_config(self):
    cce_obj = metrics.CategoricalCrossentropy(
        name='cce', dtype=dtypes.int32, label_smoothing=0.2)
    self.assertEqual(cce_obj.name, 'cce')
    self.assertEqual(cce_obj._dtype, dtypes.int32)

    old_config = cce_obj.get_config()
    self.assertAllClose(old_config['label_smoothing'], 0.2, 1e-3)

    # Check save and restore config
    cce_obj2 = metrics.CategoricalCrossentropy.from_config(old_config)
    self.assertEqual(cce_obj2.name, 'cce')
    self.assertEqual(cce_obj2._dtype, dtypes.int32)
    new_config = cce_obj2.get_config()
    self.assertDictEqual(old_config, new_config)

  def test_unweighted(self):
    cce_obj = metrics.CategoricalCrossentropy()
    self.evaluate(variables.variables_initializer(cce_obj.variables))

    y_true = np.asarray([[0, 1, 0], [0, 0, 1]])
    y_pred = np.asarray([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    result = cce_obj(y_true, y_pred)

    # EPSILON = 1e-7, y = y_true, y` = y_pred
    # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]

    # Metric = -sum(y * log(y'), axis = -1)
    #        = -((log 0.95), (log 0.1))
    #        = [0.051, 2.302]
    # Reduced metric = (0.051 + 2.302) / 2

    self.assertAllClose(self.evaluate(result), 1.176, atol=1e-3)

  def test_unweighted_from_logits(self):
    cce_obj = metrics.CategoricalCrossentropy(from_logits=True)
    self.evaluate(variables.variables_initializer(cce_obj.variables))

    y_true = np.asarray([[0, 1, 0], [0, 0, 1]])
    logits = np.asarray([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
    result = cce_obj(y_true, logits)

    # softmax = exp(logits) / sum(exp(logits), axis=-1)
    # xent = -sum(labels * log(softmax), 1)

    # exp(logits) = [[2.718, 8103.084, 1], [2.718, 2980.958, 2.718]]
    # sum(exp(logits), axis=-1) = [8106.802, 2986.394]
    # softmax = [[0.00033, 0.99954, 0.00012], [0.00091, 0.99817, 0.00091]]
    # log(softmax) = [[-8.00045, -0.00045, -9.00045],
    #                 [-7.00182, -0.00182, -7.00182]]
    # labels * log(softmax) = [[0, -0.00045, 0], [0, 0, -7.00182]]
    # xent = [0.00045, 7.00182]
    # Reduced xent = (0.00045 + 7.00182) / 2

    self.assertAllClose(self.evaluate(result), 3.5011, atol=1e-3)

  def test_weighted(self):
    cce_obj = metrics.CategoricalCrossentropy()
    self.evaluate(variables.variables_initializer(cce_obj.variables))

    y_true = np.asarray([[0, 1, 0], [0, 0, 1]])
    y_pred = np.asarray([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    sample_weight = constant_op.constant([1.5, 2.])
    result = cce_obj(y_true, y_pred, sample_weight=sample_weight)

    # EPSILON = 1e-7, y = y_true, y` = y_pred
    # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]

    # Metric = -sum(y * log(y'), axis = -1)
    #        = -((log 0.95), (log 0.1))
    #        = [0.051, 2.302]
    # Weighted metric = [0.051 * 1.5, 2.302 * 2.]
    # Reduced metric = (0.051 * 1.5 + 2.302 * 2.) / 3.5

    self.assertAllClose(self.evaluate(result), 1.338, atol=1e-3)

  def test_weighted_from_logits(self):
    cce_obj = metrics.CategoricalCrossentropy(from_logits=True)
    self.evaluate(variables.variables_initializer(cce_obj.variables))

    y_true = np.asarray([[0, 1, 0], [0, 0, 1]])
    logits = np.asarray([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
    sample_weight = constant_op.constant([1.5, 2.])
    result = cce_obj(y_true, logits, sample_weight=sample_weight)

    # softmax = exp(logits) / sum(exp(logits), axis=-1)
    # xent = -sum(labels * log(softmax), 1)
    # xent = [0.00045, 7.00182]
    # weighted xent = [0.000675, 14.00364]
    # Reduced xent = (0.000675 + 14.00364) / (1.5 + 2)

    self.assertAllClose(self.evaluate(result), 4.0012, atol=1e-3)

  def test_label_smoothing(self):
    y_true = np.asarray([[0, 1, 0], [0, 0, 1]])
    logits = np.asarray([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
    label_smoothing = 0.1

    # Label smoothing: z' = z * (1 - L) + L/n,
    #     where L = label smoothing value and n = num classes
    # Label value 1 becomes: 1 - L + L/n
    # Label value 0 becomes: L/n
    # y_true with label_smoothing = [[0.0333, 0.9333, 0.0333],
    #                               [0.0333, 0.0333, 0.9333]]

    # softmax = exp(logits) / sum(exp(logits), axis=-1)
    # xent = -sum(labels * log(softmax), 1)
    # log(softmax) = [[-8.00045, -0.00045, -9.00045],
    #                 [-7.00182, -0.00182, -7.00182]]
    # labels * log(softmax) = [[-0.26641, -0.00042, -0.29971],
    #                          [-0.23316, -0.00006, -6.53479]]
    # xent = [0.56654, 6.76801]
    # Reduced xent = (0.56654 + 6.76801) / 2

    cce_obj = metrics.CategoricalCrossentropy(
        from_logits=True, label_smoothing=label_smoothing)
    self.evaluate(variables.variables_initializer(cce_obj.variables))
    loss = cce_obj(y_true, logits)
    self.assertAllClose(self.evaluate(loss), 3.667, atol=1e-3)


def _get_model(compile_metrics):
  model_layers = [
      layers.Dense(3, activation='relu', kernel_initializer='ones'),
      layers.Dense(1, activation='sigmoid', kernel_initializer='ones')]

  model = testing_utils.get_model_from_layers(model_layers, input_shape=(4,))
  model.compile(
      loss='mae',
      metrics=compile_metrics,
      optimizer='rmsprop',
      run_eagerly=testing_utils.should_run_eagerly())
  return model


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class ResetStatesTest(keras_parameterized.TestCase):

  def test_reset_states_false_positives(self):
    fp_obj = metrics.FalsePositives()
    model = _get_model([fp_obj])
    x = np.ones((100, 4))
    y = np.zeros((100, 1))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(fp_obj.accumulator), 100.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(fp_obj.accumulator), 100.)

  def test_reset_states_false_negatives(self):
    fn_obj = metrics.FalseNegatives()
    model = _get_model([fn_obj])
    x = np.zeros((100, 4))
    y = np.ones((100, 1))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(fn_obj.accumulator), 100.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(fn_obj.accumulator), 100.)

  def test_reset_states_true_negatives(self):
    tn_obj = metrics.TrueNegatives()
    model = _get_model([tn_obj])
    x = np.zeros((100, 4))
    y = np.zeros((100, 1))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(tn_obj.accumulator), 100.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(tn_obj.accumulator), 100.)

  def test_reset_states_true_positives(self):
    tp_obj = metrics.TruePositives()
    model = _get_model([tp_obj])
    x = np.ones((100, 4))
    y = np.ones((100, 1))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(tp_obj.accumulator), 100.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(tp_obj.accumulator), 100.)

  def test_reset_states_precision(self):
    p_obj = metrics.Precision()
    model = _get_model([p_obj])
    x = np.concatenate((np.ones((50, 4)), np.ones((50, 4))))
    y = np.concatenate((np.ones((50, 1)), np.zeros((50, 1))))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(p_obj.true_positives), 50.)
    self.assertEqual(self.evaluate(p_obj.false_positives), 50.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(p_obj.true_positives), 50.)
    self.assertEqual(self.evaluate(p_obj.false_positives), 50.)

  def test_reset_states_recall(self):
    r_obj = metrics.Recall()
    model = _get_model([r_obj])
    x = np.concatenate((np.ones((50, 4)), np.zeros((50, 4))))
    y = np.concatenate((np.ones((50, 1)), np.ones((50, 1))))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(r_obj.true_positives), 50.)
    self.assertEqual(self.evaluate(r_obj.false_negatives), 50.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(r_obj.true_positives), 50.)
    self.assertEqual(self.evaluate(r_obj.false_negatives), 50.)

  def test_reset_states_sensitivity_at_specificity(self):
    s_obj = metrics.SensitivityAtSpecificity(0.5, num_thresholds=1)
    model = _get_model([s_obj])
    x = np.concatenate((np.ones((25, 4)), np.zeros((25, 4)), np.zeros((25, 4)),
                        np.ones((25, 4))))
    y = np.concatenate((np.ones((25, 1)), np.zeros((25, 1)), np.ones((25, 1)),
                        np.zeros((25, 1))))

    for _ in range(2):
      model.evaluate(x, y)
      self.assertEqual(self.evaluate(s_obj.true_positives), 25.)
      self.assertEqual(self.evaluate(s_obj.false_positives), 25.)
      self.assertEqual(self.evaluate(s_obj.false_negatives), 25.)
      self.assertEqual(self.evaluate(s_obj.true_negatives), 25.)

  def test_reset_states_specificity_at_sensitivity(self):
    s_obj = metrics.SpecificityAtSensitivity(0.5, num_thresholds=1)
    model = _get_model([s_obj])
    x = np.concatenate((np.ones((25, 4)), np.zeros((25, 4)), np.zeros((25, 4)),
                        np.ones((25, 4))))
    y = np.concatenate((np.ones((25, 1)), np.zeros((25, 1)), np.ones((25, 1)),
                        np.zeros((25, 1))))

    for _ in range(2):
      model.evaluate(x, y)
      self.assertEqual(self.evaluate(s_obj.true_positives), 25.)
      self.assertEqual(self.evaluate(s_obj.false_positives), 25.)
      self.assertEqual(self.evaluate(s_obj.false_negatives), 25.)
      self.assertEqual(self.evaluate(s_obj.true_negatives), 25.)

  def test_reset_states_auc(self):
    auc_obj = metrics.AUC(num_thresholds=3)
    model = _get_model([auc_obj])
    x = np.concatenate((np.ones((25, 4)), np.zeros((25, 4)), np.zeros((25, 4)),
                        np.ones((25, 4))))
    y = np.concatenate((np.ones((25, 1)), np.zeros((25, 1)), np.ones((25, 1)),
                        np.zeros((25, 1))))

    for _ in range(2):
      model.evaluate(x, y)
      self.assertEqual(self.evaluate(auc_obj.true_positives[1]), 25.)
      self.assertEqual(self.evaluate(auc_obj.false_positives[1]), 25.)
      self.assertEqual(self.evaluate(auc_obj.false_negatives[1]), 25.)
      self.assertEqual(self.evaluate(auc_obj.true_negatives[1]), 25.)

  def test_reset_states_mean_iou(self):
    m_obj = metrics.MeanIoU(num_classes=2)
    model = _get_model([m_obj])
    x = np.asarray([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
                   dtype=np.float32)
    y = np.asarray([[0], [1], [1], [1]], dtype=np.float32)
    model.evaluate(x, y)
    self.assertArrayNear(self.evaluate(m_obj.total_cm)[0], [1, 0], 1e-1)
    self.assertArrayNear(self.evaluate(m_obj.total_cm)[1], [3, 0], 1e-1)
    model.evaluate(x, y)
    self.assertArrayNear(self.evaluate(m_obj.total_cm)[0], [1, 0], 1e-1)
    self.assertArrayNear(self.evaluate(m_obj.total_cm)[1], [3, 0], 1e-1)


if __name__ == '__main__':
  test.main()
