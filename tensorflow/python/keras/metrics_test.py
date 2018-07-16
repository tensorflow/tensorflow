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

import numpy as np

from tensorflow.python import keras
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import test


class KerasMetricsTest(test.TestCase):

  def test_metrics(self):
    with self.test_session():
      y_a = keras.backend.variable(np.random.random((6, 7)))
      y_b = keras.backend.variable(np.random.random((6, 7)))
      for metric in [keras.metrics.binary_accuracy,
                     keras.metrics.categorical_accuracy]:
        output = metric(y_a, y_b)
        self.assertEqual(keras.backend.eval(output).shape, (6,))

  def test_sparse_categorical_accuracy(self):
    with self.test_session():
      metric = keras.metrics.sparse_categorical_accuracy
      y_a = keras.backend.variable(np.random.randint(0, 7, (6,)))
      y_b = keras.backend.variable(np.random.random((6, 7)))
      self.assertEqual(keras.backend.eval(metric(y_a, y_b)).shape, (6,))

  def test_sparse_top_k_categorical_accuracy(self):
    with self.test_session():
      y_pred = keras.backend.variable(np.array([[0.3, 0.2, 0.1],
                                                [0.1, 0.2, 0.7]]))
      y_true = keras.backend.variable(np.array([[1], [0]]))
      result = keras.backend.eval(
          keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=3))
      self.assertEqual(result, 1)
      result = keras.backend.eval(
          keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=2))
      self.assertEqual(result, 0.5)
      result = keras.backend.eval(
          keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=1))
      self.assertEqual(result, 0.)

  def test_top_k_categorical_accuracy(self):
    with self.test_session():
      y_pred = keras.backend.variable(np.array([[0.3, 0.2, 0.1],
                                                [0.1, 0.2, 0.7]]))
      y_true = keras.backend.variable(np.array([[0, 1, 0], [1, 0, 0]]))
      result = keras.backend.eval(
          keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3))
      self.assertEqual(result, 1)
      result = keras.backend.eval(
          keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2))
      self.assertEqual(result, 0.5)
      result = keras.backend.eval(
          keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=1))
      self.assertEqual(result, 0.)

  def test_stateful_metrics(self):
    with self.test_session():
      np.random.seed(1334)

      class BinaryTruePositives(keras.layers.Layer):
        """Stateful Metric to count the total true positives over all batches.

        Assumes predictions and targets of shape `(samples, 1)`.

        Arguments:
            threshold: Float, lower limit on prediction value that counts as a
                positive class prediction.
            name: String, name for the metric.
        """

        def __init__(self, name='true_positives', **kwargs):
          super(BinaryTruePositives, self).__init__(name=name, **kwargs)
          self.true_positives = keras.backend.variable(value=0, dtype='int32')
          self.stateful = True

        def reset_states(self):
          keras.backend.set_value(self.true_positives, 0)

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
      config = keras.metrics.serialize(metric_fn)
      metric_fn = keras.metrics.deserialize(
          config, custom_objects={'BinaryTruePositives': BinaryTruePositives})

      # Test on simple model
      inputs = keras.Input(shape=(2,))
      outputs = keras.layers.Dense(1, activation='sigmoid')(inputs)
      model = keras.Model(inputs, outputs)
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


if __name__ == '__main__':
  test.main()
