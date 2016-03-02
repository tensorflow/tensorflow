# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Tests for SdcaModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf
from tensorflow.contrib.linear_optimizer.python.ops.sdca_ops import SdcaModel
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.platform import googletest


def make_example_proto(feature_dict, target, value=1.0):
  e = tf.train.Example()
  features = e.features

  features.feature['target'].float_list.value.append(target)

  for key, values in feature_dict.iteritems():
    features.feature[key + '_indices'].int64_list.value.extend(values)
    features.feature[key + '_values'].float_list.value.extend([value] *
                                                              len(values))

  return e


def make_example_dict(example_protos, example_weights):

  def parse_examples(example_protos):
    features = {
        'target': tf.FixedLenFeature(shape=[1],
                                     dtype=tf.float32,
                                     default_value=0),
        'age_indices': tf.VarLenFeature(dtype=tf.int64),
        'age_values': tf.VarLenFeature(dtype=tf.float32),
        'gender_indices': tf.VarLenFeature(dtype=tf.int64),
        'gender_values': tf.VarLenFeature(dtype=tf.float32)
    }
    return tf.parse_example(
        [e.SerializeToString() for e in example_protos], features)

  sparse_merge = lambda ids, values: tf.sparse_merge(ids, values, ids.shape[1])

  parsed = parse_examples(example_protos)
  sparse_features = [
      sparse_merge(parsed['age_indices'], parsed['age_values']),
      sparse_merge(parsed['gender_indices'], parsed['gender_values'])
  ]
  return dict(sparse_features=sparse_features,
              dense_features=[],
              example_weights=example_weights,
              example_labels=tf.reshape(parsed['target'], [-1]),
              example_ids=['%d' % i for i in xrange(0, len(example_protos))])


def make_variable_dict(max_age, max_gender):
  # TODO(dbaylor):  Figure out how to derive max_age & max_gender from
  # examples_dict.
  age_weights = tf.Variable(tf.zeros([max_age + 1], dtype=tf.float32))
  gender_weights = tf.Variable(tf.zeros([max_gender + 1], dtype=tf.float32))
  primal_loss = tf.Variable(tf.zeros([], dtype=tf.float64))
  return dict(sparse_features_weights=[age_weights, gender_weights],
              dense_features_weights=[],
              primal_loss=primal_loss)


# Setup the single container shared across all tests. This is testing proper
# isolation across optimizers instantiated in each of the tests below.
CONTAINER = uuid.uuid4().hex


# Clear the shared container.
def tearDown():
  # TODO(katsiapis): Proper cleanup of Containers when possible.
  pass


class SdcaOptimizerTest(TensorFlowTestCase):

  def _single_threaded_test_session(self):
    config = tf.ConfigProto(inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    return self.test_session(use_gpu=False, config=config)

  def testSimpleLogistic(self):
    # Setup test data
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 1),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=0.5,
                     symmetric_l1_regularization=0,
                     loss_type='logistic_loss',
                     prior=0.0)
      tf.initialize_all_variables().run()
      lr = SdcaModel(CONTAINER, examples, variables, options)
      unregularized_loss = lr.unregularized_loss(examples)
      loss = lr.regularized_loss(examples)
      prediction = lr.predictions(examples)
      self.assertAllClose(0.693147, unregularized_loss.eval())
      self.assertAllClose(0.693147, loss.eval())
      lr.minimize().run()
      self.assertAllClose(0.395226, unregularized_loss.eval(),
                          rtol=3e-2, atol=3e-2)
      self.assertAllClose(0.657446, loss.eval(),
                          rtol=3e-2, atol=3e-2)
      predicted_labels = tf.cast(
          tf.greater_equal(prediction,
                           tf.ones_like(prediction) * 0.5), tf.float32)
      self.assertAllEqual([0, 1], predicted_labels.eval())

  def testSomeUnweightedExamples(self):
    # Setup test data with 4 examples, but should produce the same
    # results as testSimple.
    example_protos = [
        # Will be used.
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0),
        # Will be ignored.
        make_example_proto(
            {'age': [1],
             'gender': [0]}, 0),
        # Will be used.
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 1),
        # Will be ignored.
        make_example_proto(
            {'age': [1],
             'gender': [0]}, 1),
    ]
    example_weights = [1.0, 0.0, 1.0, 0.0]
    with self._single_threaded_test_session():
      # Only use examples 0 and 2
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=0.5,
                     symmetric_l1_regularization=0,
                     loss_type='logistic_loss')
      tf.initialize_all_variables().run()
      lr = SdcaModel(CONTAINER, examples, variables, options)
      unregularized_loss = lr.unregularized_loss(examples)
      loss = lr.regularized_loss(examples)
      prediction = lr.predictions(examples)
      lr.minimize().run()
      self.assertAllClose(0.395226, unregularized_loss.eval(),
                          rtol=3e-2, atol=3e-2)
      self.assertAllClose(0.657446, loss.eval(),
                          rtol=3e-2, atol=3e-2)
      predicted_labels = tf.cast(
          tf.greater_equal(prediction,
                           tf.ones_like(prediction) * 0.5), tf.float32)
      self.assertAllClose([0, 1, 1, 1], predicted_labels.eval())

  def testNoWeightedExamples(self):
    # Setup test data with 1 positive, and 1 negative example.
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 1),
    ]
    # Zeroed out example weights.
    example_weights = [0.0, 0.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=0.5,
                     symmetric_l1_regularization=0,
                     loss_type='logistic_loss')
      tf.initialize_all_variables().run()
      lr = SdcaModel(CONTAINER, examples, variables, options)
      self.assertAllClose([0.5, 0.5], lr.predictions(examples).eval())
      with self.assertRaisesOpError(
          'No weighted examples in 2 training examples'):
        lr.minimize().run()
      self.assertAllClose([0.5, 0.5], lr.predictions(examples).eval())

  def testImbalanced(self):
    # Setup test data with 1 positive, and 3 negative examples.
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [2],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [3],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 1),
    ]
    example_weights = [1.0, 1.0, 1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(3, 1)
      options = dict(symmetric_l2_regularization=0.25,
                     symmetric_l1_regularization=0,
                     loss_type='logistic_loss',
                     prior=-1.09861)
      tf.initialize_all_variables().run()
      lr = SdcaModel(CONTAINER, examples, variables, options)
      unregularized_loss = lr.unregularized_loss(examples)
      loss = lr.regularized_loss(examples)
      prediction = lr.predictions(examples)
      lr.minimize().run()
      self.assertAllClose(0.331710, unregularized_loss.eval(),
                          rtol=3e-2, atol=3e-2)
      self.assertAllClose(0.591295, loss.eval(), rtol=3e-2, atol=3e-2)
      predicted_labels = tf.cast(
          tf.greater_equal(prediction,
                           tf.ones_like(prediction) * 0.5), tf.float32)
      self.assertAllEqual([0, 0, 0, 1], predicted_labels.eval())

  def testImbalancedWithExampleWeights(self):
    # Setup test data with 1 positive, and 3 negative examples.
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 1),
    ]
    example_weights = [3.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=0.25,
                     symmetric_l1_regularization=0,
                     loss_type='logistic_loss')
      tf.initialize_all_variables().run()
      lr = SdcaModel(CONTAINER, examples, variables, options)
      unregularized_loss = lr.unregularized_loss(examples)
      loss = lr.regularized_loss(examples)
      prediction = lr.predictions(examples)
      lr.minimize().run()
      self.assertAllClose(0.266189, unregularized_loss.eval(),
                          rtol=3e-2, atol=3e-2)
      self.assertAllClose(0.571912, loss.eval(), rtol=3e-2, atol=3e-2)
      predicted_labels = tf.cast(
          tf.greater_equal(prediction,
                           tf.ones_like(prediction) * 0.5), tf.float32)
      self.assertAllEqual([0, 1], predicted_labels.eval())

  def testSimpleLinear(self):
    # Setup test data
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, -10.0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 14.0),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=0.5,
                     symmetric_l1_regularization=0,
                     loss_type='squared_loss',
                     prior=0.0)
      tf.initialize_all_variables().run()
      lr = SdcaModel(CONTAINER, examples, variables, options)
      prediction = lr.predictions(examples)

      lr.minimize().run()

      # Predictions should be 2/3 of label due to minimizing regularized loss:
      #   (label - 2 * weight)^2 / 2 + L2 * 2 * weight^2
      self.assertAllClose([-20.0 / 3.0, 28.0 / 3.0],
                          prediction.eval(),
                          rtol=0.005)

  def testLinearRegularization(self):
    # Setup test data
    example_protos = [
        # 2 identical examples
        make_example_proto(
            {'age': [0],
             'gender': [0]}, -10.0),
        make_example_proto(
            {'age': [0],
             'gender': [0]}, -10.0),
        # 2 more identical examples
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 14.0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 14.0),
    ]
    example_weights = [1.0, 1.0, 1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=4.0,
                     symmetric_l1_regularization=0,
                     loss_type='squared_loss',
                     prior=0.0)
      tf.initialize_all_variables().run()
      lr = SdcaModel(CONTAINER, examples, variables, options)
      prediction = lr.predictions(examples)

      lr.minimize().run()

      # Predictions should be 1/5 of label due to minimizing regularized loss:
      #   (label - 2 * weight)^2 + L2 * 16 * weight^2
      optimal1 = -10.0 / 5.0
      optimal2 = 14.0 / 5.0
      self.assertAllClose(
          [optimal1, optimal1, optimal2, optimal2],
          prediction.eval(),
          rtol=0.01)

  def testLinearFeatureValues(self):
    # Setup test data
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, -10.0, -2.0),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 14.0, 2.0),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)

      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=0.5,
                     symmetric_l1_regularization=0,
                     loss_type='squared_loss',
                     prior=0.0)
      tf.initialize_all_variables().run()
      lr = SdcaModel(CONTAINER, examples, variables, options)
      prediction = lr.predictions(examples)

      lr.minimize().run()

      # Predictions should be 8/9 of label due to minimizing regularized loss:
      #   (label - 2 * 2 * weight)^2 / 2 + L2 * 2 * weight^2
      self.assertAllClose([-10.0 * 8 / 9, 14.0 * 8 / 9],
                          prediction.eval(),
                          rtol=0.07)

  def testLinearDenseFeatures(self):
    with self._single_threaded_test_session():
      examples = dict(sparse_features=[],
                      dense_features=[tf.convert_to_tensor(
                          [-2.0, 0.0],
                          dtype=tf.float32), tf.convert_to_tensor(
                              [0.0, 2.0],
                              dtype=tf.float32)],
                      example_weights=[1.0, 1.0],
                      example_labels=[-10.0, 14.0],
                      example_ids=['%d' % i for i in xrange(0, 2)])
      variables = dict(sparse_features_weights=[],
                       dense_features_weights=[tf.Variable(tf.zeros(
                           [1],
                           dtype=tf.float32)), tf.Variable(tf.zeros(
                               [1],
                               dtype=tf.float32))],
                       dual=tf.Variable(tf.zeros(
                           [2],
                           dtype=tf.float32)),
                       primal_loss=tf.Variable(tf.zeros(
                           [],
                           dtype=tf.float64)))
      options = dict(symmetric_l2_regularization=0.5,
                     symmetric_l1_regularization=0,
                     loss_type='squared_loss',
                     prior=0.0)
      tf.initialize_all_variables().run()
      lr = SdcaModel(CONTAINER, examples, variables, options)
      prediction = lr.predictions(examples)

      lr.minimize().run()

      # Predictions should be 4/5 of label due to minimizing regularized loss:
      #   (label - 2 * weight)^2 / 2 + L2 * weight^2
      self.assertAllClose([-10.0 * 4 / 5, 14.0 * 4 / 5],
                          prediction.eval(),
                          rtol=0.01)

      loss = lr.regularized_loss(examples)
      self.assertAllClose(
          (4.0 + 7.84 + 16.0 + 31.36) / 2,
          loss.eval(),
          rtol=0.01)

if __name__ == '__main__':
  googletest.main()
