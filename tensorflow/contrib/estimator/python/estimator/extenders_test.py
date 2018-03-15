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

from tensorflow.contrib.estimator.python.estimator import extenders
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import estimator_lib
from tensorflow.python.estimator.canned import linear
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
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
      self.assertTrue(isinstance(config, estimator_lib.RunConfig))
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
      self.assertTrue(isinstance(config, estimator_lib.RunConfig))
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


class ForwardFeaturesTest(test.TestCase):
  """Tests forward_features."""

  def test_forward_single_key(self):

    def input_fn():
      return {'x': [[3.], [5.]], 'id': [[101], [102]]}, [[1.], [2.]]

    estimator = linear.LinearRegressor([fc.numeric_column('x')])
    estimator.train(input_fn=input_fn, steps=1)

    self.assertNotIn('id', next(estimator.predict(input_fn=input_fn)))
    estimator = extenders.forward_features(estimator, 'id')
    predictions = next(estimator.predict(input_fn=input_fn))
    self.assertIn('id', predictions)
    self.assertEqual(101, predictions['id'])

  def test_forward_list(self):

    def input_fn():
      return {'x': [[3.], [5.]], 'id': [[101], [102]]}, [[1.], [2.]]

    estimator = linear.LinearRegressor([fc.numeric_column('x')])
    estimator.train(input_fn=input_fn, steps=1)

    self.assertNotIn('id', next(estimator.predict(input_fn=input_fn)))
    estimator = extenders.forward_features(estimator, ['x', 'id'])
    predictions = next(estimator.predict(input_fn=input_fn))
    self.assertIn('id', predictions)
    self.assertIn('x', predictions)
    self.assertEqual(101, predictions['id'])
    self.assertEqual(3., predictions['x'])

  def test_forward_all(self):

    def input_fn():
      return {'x': [[3.], [5.]], 'id': [[101], [102]]}, [[1.], [2.]]

    estimator = linear.LinearRegressor([fc.numeric_column('x')])
    estimator.train(input_fn=input_fn, steps=1)

    self.assertNotIn('id', next(estimator.predict(input_fn=input_fn)))
    self.assertNotIn('x', next(estimator.predict(input_fn=input_fn)))
    estimator = extenders.forward_features(estimator)
    predictions = next(estimator.predict(input_fn=input_fn))
    self.assertIn('id', predictions)
    self.assertIn('x', predictions)
    self.assertEqual(101, predictions['id'])
    self.assertEqual(3., predictions['x'])

  def test_key_should_be_string(self):
    estimator = linear.LinearRegressor([fc.numeric_column('x')])
    with self.assertRaisesRegexp(TypeError, 'keys should be either a string'):
      extenders.forward_features(estimator, estimator)

  def test_key_should_be_list_of_string(self):
    estimator = linear.LinearRegressor([fc.numeric_column('x')])
    with self.assertRaisesRegexp(TypeError, 'should be a string'):
      extenders.forward_features(estimator, ['x', estimator])

  def test_key_should_be_in_features(self):

    def input_fn():
      return {'x': [[3.], [5.]], 'id': [[101], [102]]}, [[1.], [2.]]

    estimator = linear.LinearRegressor([fc.numeric_column('x')])
    estimator.train(input_fn=input_fn, steps=1)

    estimator = extenders.forward_features(estimator, 'y')
    with self.assertRaisesRegexp(ValueError,
                                 'keys should be exist in features'):
      next(estimator.predict(input_fn=input_fn))

  def test_forwarded_feature_should_not_be_a_sparse_tensor(self):

    def input_fn():
      return {
          'x': [[3.], [5.]],
          'id':
              sparse_tensor.SparseTensor(
                  values=['1', '2'],
                  indices=[[0, 0], [1, 0]],
                  dense_shape=[2, 1])
      }, [[1.], [2.]]

    estimator = linear.LinearRegressor([fc.numeric_column('x')])
    estimator.train(input_fn=input_fn, steps=1)

    estimator = extenders.forward_features(estimator)
    with self.assertRaisesRegexp(ValueError,
                                 'Forwarded feature.* should be a Tensor.'):
      next(estimator.predict(input_fn=input_fn))

  def test_predictions_should_be_dict(self):

    def input_fn():
      return {'x': [[3.], [5.]], 'id': [[101], [102]]}

    def model_fn(features, mode):
      del features
      global_step = training.get_global_step()
      return estimator_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant([5.]),
          predictions=constant_op.constant([5.]),
          train_op=global_step.assign_add(1))

    estimator = estimator_lib.Estimator(model_fn=model_fn)
    estimator.train(input_fn=input_fn, steps=1)

    estimator = extenders.forward_features(estimator)
    with self.assertRaisesRegexp(ValueError, 'Predictions should be a dict'):
      next(estimator.predict(input_fn=input_fn))

  def test_should_not_conflict_with_existing_predictions(self):

    def input_fn():
      return {'x': [[3.], [5.]], 'id': [[101], [102]]}

    def model_fn(features, mode):
      del features
      global_step = training.get_global_step()
      return estimator_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant([5.]),
          predictions={'x': constant_op.constant([5.])},
          train_op=global_step.assign_add(1))

    estimator = estimator_lib.Estimator(model_fn=model_fn)
    estimator.train(input_fn=input_fn, steps=1)

    estimator = extenders.forward_features(estimator)
    with self.assertRaisesRegexp(ValueError, 'Cannot forward feature key'):
      next(estimator.predict(input_fn=input_fn))


if __name__ == '__main__':
  test.main()
