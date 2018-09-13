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

import os
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import metrics
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.checkpointable import util as checkpointable_utils


class KerasMetricsTest(test.TestCase):

  def test_metrics(self):
    with self.cached_session():
      y_a = K.variable(np.random.random((6, 7)))
      y_b = K.variable(np.random.random((6, 7)))
      for metric in [metrics.binary_accuracy, metrics.categorical_accuracy]:
        output = metric(y_a, y_b)
        self.assertEqual(K.eval(output).shape, (6,))

  def test_sparse_categorical_accuracy(self):
    with self.cached_session():
      metric = metrics.sparse_categorical_accuracy
      y_true = K.variable(np.random.randint(0, 7, (6,)))
      y_pred = K.variable(np.random.random((6, 7)))
      self.assertEqual(K.eval(metric(y_true, y_pred)).shape, (6,))

  def test_sparse_categorical_accuracy_float(self):
    with self.cached_session():
      metric = metrics.sparse_categorical_accuracy
      y_true = K.variable(np.random.random((6,)))
      y_pred = K.variable(np.random.random((6, 7)))
      self.assertEqual(K.eval(metric(y_true, y_pred)).shape, (6,))

  def test_sparse_categorical_accuracy_eager(self):
    """Tests that ints passed in via Eager return results. See b/113504761."""
    with context.eager_mode():
      metric = metrics.sparse_categorical_accuracy
      y_true = np.arange(6).reshape([6, 1])
      y_pred = np.arange(36).reshape([6, 6])
      self.assertAllEqual(metric(y_true, y_pred), [0., 0., 0., 0., 0., 1.])

  def test_sparse_categorical_accuracy_float_eager(self):
    """Tests that floats passed in via Eager return results. See b/113504761."""
    with context.eager_mode():
      metric = metrics.sparse_categorical_accuracy
      y_true = np.arange(6, dtype=np.float32).reshape([6, 1])
      y_pred = np.arange(36).reshape([6, 6])
      self.assertAllEqual(metric(y_true, y_pred), [0., 0., 0., 0., 0., 1.])

  def test_sparse_top_k_categorical_accuracy(self):
    with self.cached_session():
      y_pred = K.variable(np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.7]]))
      y_true = K.variable(np.array([[1], [0]]))
      result = K.eval(
          metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=3))
      self.assertEqual(result, 1)
      result = K.eval(
          metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=2))
      self.assertEqual(result, 0.5)
      result = K.eval(
          metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=1))
      self.assertEqual(result, 0.)

  def test_top_k_categorical_accuracy(self):
    with self.cached_session():
      y_pred = K.variable(np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.7]]))
      y_true = K.variable(np.array([[0, 1, 0], [1, 0, 0]]))
      result = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred, k=3))
      self.assertEqual(result, 1)
      result = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred, k=2))
      self.assertEqual(result, 0.5)
      result = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred, k=1))
      self.assertEqual(result, 0.)

  def test_stateful_metrics(self):
    with self.cached_session():
      np.random.seed(1334)

      class BinaryTruePositives(layers.Layer):
        """Stateful Metric to count the total true positives over all batches.

        Assumes predictions and targets of shape `(samples, 1)`.

        Arguments:
            threshold: Float, lower limit on prediction value that counts as a
                positive class prediction.
            name: String, name for the metric.
        """

        def __init__(self, name='true_positives', **kwargs):
          super(BinaryTruePositives, self).__init__(name=name, **kwargs)
          self.true_positives = K.variable(value=0, dtype='int32')
          self.stateful = True

        def reset_states(self):
          K.set_value(self.true_positives, 0)

        def __call__(self, y_true, y_pred):
          """Computes the number of true positives in a batch.

          Args:
              y_true: Tensor, batch_wise labels
              y_pred: Tensor, batch_wise predictions

          Returns:
              The total number of true positives seen this epoch at the
                  completion of the batch.
          """
          y_true = math_ops.cast(y_true, 'int32')
          y_pred = math_ops.cast(math_ops.round(y_pred), 'int32')
          correct_preds = math_ops.cast(math_ops.equal(y_pred, y_true), 'int32')
          true_pos = math_ops.cast(
              math_ops.reduce_sum(correct_preds * y_true), 'int32')
          current_true_pos = self.true_positives * 1
          self.add_update(
              state_ops.assign_add(self.true_positives, true_pos),
              inputs=[y_true, y_pred])
          return current_true_pos + true_pos

      metric_fn = BinaryTruePositives()
      config = metrics.serialize(metric_fn)
      metric_fn = metrics.deserialize(
          config, custom_objects={'BinaryTruePositives': BinaryTruePositives})

      # Test on simple model
      inputs = layers.Input(shape=(2,))
      outputs = layers.Dense(1, activation='sigmoid')(inputs)
      model = Model(inputs, outputs)
      model.compile(optimizer='sgd',
                    loss='binary_crossentropy',
                    metrics=['acc', metric_fn])

      # Test fit, evaluate
      samples = 100
      x = np.random.random((samples, 2))
      y = np.random.randint(2, size=(samples, 1))
      val_samples = 10
      val_x = np.random.random((val_samples, 2))
      val_y = np.random.randint(2, size=(val_samples, 1))

      history = model.fit(x, y,
                          epochs=1,
                          batch_size=10,
                          validation_data=(val_x, val_y))
      outs = model.evaluate(x, y, batch_size=10)
      preds = model.predict(x)

      def ref_true_pos(y_true, y_pred):
        return np.sum(np.logical_and(y_pred > 0.5, y_true == 1))

      # Test correctness (e.g. updates should have been run)
      self.assertAllClose(outs[2], ref_true_pos(y, preds), atol=1e-5)

      # Test correctness of the validation metric computation
      val_preds = model.predict(val_x)
      val_outs = model.evaluate(val_x, val_y, batch_size=10)
      self.assertAllClose(
          val_outs[2], ref_true_pos(val_y, val_preds), atol=1e-5)
      self.assertAllClose(
          val_outs[2], history.history['val_true_positives'][-1], atol=1e-5)

      # Test with generators
      gen = [(np.array([x0]), np.array([y0])) for x0, y0 in zip(x, y)]
      val_gen = [(np.array([x0]), np.array([y0]))
                 for x0, y0 in zip(val_x, val_y)]
      history = model.fit_generator(iter(gen),
                                    epochs=1,
                                    steps_per_epoch=samples,
                                    validation_data=iter(val_gen),
                                    validation_steps=val_samples)
      outs = model.evaluate_generator(iter(gen), steps=samples)
      preds = model.predict_generator(iter(gen), steps=samples)

      # Test correctness of the metric results
      self.assertAllClose(outs[2], ref_true_pos(y, preds), atol=1e-5)

      # Test correctness of the validation metric computation
      val_preds = model.predict_generator(iter(val_gen), steps=val_samples)
      val_outs = model.evaluate_generator(iter(val_gen), steps=val_samples)
      self.assertAllClose(
          val_outs[2], ref_true_pos(val_y, val_preds), atol=1e-5)
      self.assertAllClose(
          val_outs[2], history.history['val_true_positives'][-1], atol=1e-5)

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
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

  @test_util.run_in_graph_and_eager_modes
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
      sess.run(variables.variables_initializer(m.variables))

      # check __call__()
      result_t = m(v, sample_weight=w)
      result = sess.run(result_t, feed_dict=({v: 100, w: 0.5}))
      self.assertEqual(sess.run(m.total), 50)
      self.assertEqual(sess.run(m.count), 0.5)
      self.assertEqual(result, 50 / 0.5)

      # check update_state() and result()
      result = sess.run(result_t, feed_dict=({v: [1, 5], w: [1, 0.2]}))
      self.assertAlmostEqual(sess.run(m.total), 52, 2)  # 50 + 1 + 5 * 0.2
      self.assertAlmostEqual(sess.run(m.count), 1.7, 2)  # 0.5 + 1.2
      self.assertAlmostEqual(result, 52 / 1.7, 2)

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
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

    # check incompatible shapes
    with self.assertRaisesRegexp(ValueError,
                                 r'Shapes \(1,\) and \(2,\) are incompatible'):
      acc_obj.update_state([1, 1], [1])

  @test_util.run_in_graph_and_eager_modes
  def test_binary_accuracy_threshold(self):
    acc_obj = metrics.BinaryAccuracy(threshold=0.7)
    self.evaluate(variables.variables_initializer(acc_obj.variables))
    result_t = acc_obj([[1], [1], [0], [0]], [[0.9], [0.6], [0.4], [0.8]])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.5, 2)

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_result(self):

    class InvalidResult(metrics.Metric):

      def __init__(self, name='invalid-result', dtype=dtypes.float64):
        super(InvalidResult, self).__init__(name=name, dtype=dtype)

      def update_state(self, *args, **kwargs):
        pass

      def result(self):
        return 1

    invalid_result_obj = InvalidResult()
    with self.assertRaisesRegexp(
        TypeError,
        'Metric invalid-result\'s result must be a Tensor or Operation, given:'
    ):
      invalid_result_obj.result()

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_update(self):

    class InvalidUpdate(metrics.Metric):

      def __init__(self, name='invalid-update', dtype=dtypes.float64):
        super(InvalidUpdate, self).__init__(name=name, dtype=dtype)

      def update_state(self, *args, **kwargs):
        return [1]

      def result(self):
        pass

    invalid_update_obj = InvalidUpdate()
    with self.assertRaisesRegexp(
        TypeError,
        'Metric invalid-update\'s update must be a Tensor or Operation, given:'
    ):
      invalid_update_obj.update_state()


if __name__ == '__main__':
  test.main()
