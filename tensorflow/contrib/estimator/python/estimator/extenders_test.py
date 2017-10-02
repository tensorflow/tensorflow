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
"""extenders tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.contrib.estimator.python.estimator import extenders
from tensorflow.python.estimator import run_config
from tensorflow.python.estimator.canned import linear
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import training


def get_input_fn(x, y):

  def input_fn():
    dataset = dataset_ops.Dataset.from_tensor_slices({'x': x, 'y': y})
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    labels = features.pop('y')
    return features, labels

  return input_fn


class AddMetricsTest(test.TestCase):

  def test_should_add_metrics(self):
    input_fn = get_input_fn(
        x=np.arange(4)[:, None, None], y=np.ones(4)[:, None])
    estimator = linear.LinearClassifier([fc.numeric_column('x')])

    def metric_fn(features):
      return {'mean_x': metrics_lib.mean(features['x'])}

    estimator = extenders.add_metrics(estimator, metric_fn)

    estimator.train(input_fn=input_fn)
    metrics = estimator.evaluate(input_fn=input_fn)
    self.assertIn('mean_x', metrics)
    self.assertEqual(1.5, metrics['mean_x'])
    # assert that it keeps original estimators metrics
    self.assertIn('auc', metrics)

  def test_should_error_out_for_not_recognized_args(self):
    estimator = linear.LinearClassifier([fc.numeric_column('x')])

    def metric_fn(features, not_recognized):
      _, _ = features, not_recognized
      return {}

    with self.assertRaisesRegexp(ValueError, 'not_recognized'):
      estimator = extenders.add_metrics(estimator, metric_fn)

  def test_all_supported_args(self):
    input_fn = get_input_fn(x=[[[0.]]], y=[[[1]]])
    estimator = linear.LinearClassifier([fc.numeric_column('x')])

    def metric_fn(features, predictions, labels, config):
      self.assertIn('x', features)
      self.assertIsNotNone(labels)
      self.assertIn('logistic', predictions)
      self.assertTrue(isinstance(config, run_config.RunConfig))
      return {}

    estimator = extenders.add_metrics(estimator, metric_fn)

    estimator.train(input_fn=input_fn)
    estimator.evaluate(input_fn=input_fn)

  def test_all_supported_args_in_different_order(self):
    input_fn = get_input_fn(x=[[[0.]]], y=[[[1]]])
    estimator = linear.LinearClassifier([fc.numeric_column('x')])

    def metric_fn(labels, config, features, predictions):
      self.assertIn('x', features)
      self.assertIsNotNone(labels)
      self.assertIn('logistic', predictions)
      self.assertTrue(isinstance(config, run_config.RunConfig))
      return {}

    estimator = extenders.add_metrics(estimator, metric_fn)

    estimator.train(input_fn=input_fn)
    estimator.evaluate(input_fn=input_fn)

  def test_all_args_are_optional(self):
    input_fn = get_input_fn(x=[[[0.]]], y=[[[1]]])
    estimator = linear.LinearClassifier([fc.numeric_column('x')])

    def metric_fn():
      return {'two': metrics_lib.mean(constant_op.constant([2.]))}

    estimator = extenders.add_metrics(estimator, metric_fn)

    estimator.train(input_fn=input_fn)
    metrics = estimator.evaluate(input_fn=input_fn)
    self.assertEqual(2., metrics['two'])

  def test_overrides_existing_metrics(self):
    input_fn = get_input_fn(x=[[[0.]]], y=[[[1]]])
    estimator = linear.LinearClassifier([fc.numeric_column('x')])
    estimator.train(input_fn=input_fn)
    metrics = estimator.evaluate(input_fn=input_fn)
    self.assertNotEqual(2., metrics['auc'])

    def metric_fn():
      return {'auc': metrics_lib.mean(constant_op.constant([2.]))}

    estimator = extenders.add_metrics(estimator, metric_fn)
    metrics = estimator.evaluate(input_fn=input_fn)
    self.assertEqual(2., metrics['auc'])


class ClipGradientsByNormTest(test.TestCase):
  """Tests clip_gradients_by_norm."""

  def test_applies_norm(self):
    optimizer = extenders.clip_gradients_by_norm(
        training.GradientDescentOptimizer(1.0), clip_norm=3.)
    with ops.Graph().as_default():
      w = variables.Variable(1., name='weight')
      x = constant_op.constant(5.)
      y = -x * w
      grads = optimizer.compute_gradients(y, var_list=[w])[0]
      opt_op = optimizer.minimize(y, var_list=[w])
      with training.MonitoredSession() as sess:
        grads_value = sess.run(grads)
        self.assertEqual(-5., grads_value[0])
        sess.run(opt_op)
        new_w = sess.run(w)
        self.assertEqual(4., new_w)  # 1 + 1*3 (w - lr * clipped_grad)

  def test_name(self):
    optimizer = extenders.clip_gradients_by_norm(
        training.GradientDescentOptimizer(1.0), clip_norm=3.)
    self.assertEqual('ClipByNormGradientDescent', optimizer.get_name())


if __name__ == '__main__':
  test.main()
