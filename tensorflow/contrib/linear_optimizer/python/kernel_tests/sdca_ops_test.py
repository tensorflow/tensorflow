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


def make_example_proto(feature_dict, target):
  e = tf.train.Example()
  features = e.features

  features.feature['target'].float_list.value.append(target)

  for key, values in feature_dict.iteritems():
    features.feature[key + '_indices'].int64_list.value.extend(values)
    features.feature[key + '_values'].float_list.value.extend([1.0] *
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

  # TODO(rohananil): This converts two sparse tensors, into one sparse feature
  # tensor. Use the tf core op once its merged in.
  def sf_from_st(ids, weights):
    example_indices, _ = tf.split(1, 2, ids.indices)
    feature_indices = tf.expand_dims(ids.values, 1)
    indices = tf.concat(1, [example_indices, feature_indices])
    return tf.SparseTensor(indices, weights.values, ids.shape)

  parsed = parse_examples(example_protos)
  return dict(sparse_features=[
      sf_from_st(parsed['age_indices'], parsed['age_values']), sf_from_st(
          parsed['gender_indices'], parsed['gender_values'])
  ],
              dense_features=[],
              example_weights=example_weights,
              example_labels=tf.reshape(parsed['target'], [-1]),
              example_ids=['%d' % i for i in xrange(0, len(example_protos))])


def make_variable_dict(max_age, max_gender):
  # TODO(dbaylor):  Figure out how to derive max_age & max_gender from
  # examples_dict.
  age_weights = tf.Variable(tf.zeros([max_age + 1], dtype=tf.float32))
  gender_weights = tf.Variable(tf.zeros([max_gender + 1], dtype=tf.float32))
  training_log_loss = tf.Variable(tf.zeros([], dtype=tf.float64))
  return dict(sparse_features_weights=[age_weights, gender_weights],
              dense_features_weights=[],
              training_log_loss=training_log_loss)


# Setup the single container shared across all tests. This is testing proper
# isolation across optimizers instantiated in each of the tests below.
CONTAINER = uuid.uuid4().hex


# Clear the shared container.
def tearDown():
  # TODO(katsiapis): Proper cleanup of Containers when possible.
  pass


class SdcaOptimizerTest(TensorFlowTestCase):

  def testSimple(self):
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
    with self.test_session(use_gpu=False):
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
      self.assertAllClose(0.395226, unregularized_loss.eval())
      self.assertAllClose(0.657446, loss.eval())
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
    with self.test_session(use_gpu=False):
      # Only use examples 0 and 2
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
      self.assertAllClose(0.395226, unregularized_loss.eval())
      self.assertAllClose(0.526336, loss.eval())
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
    with self.test_session(use_gpu=False):
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
    with self.test_session(use_gpu=False):
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
      self.assertAllClose(0.331710,
                          unregularized_loss.eval(),
                          rtol=3e-2,
                          atol=3e-2)
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
    with self.test_session(use_gpu=False):
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
      self.assertAllClose(0.266189,
                          unregularized_loss.eval(),
                          rtol=3e-2,
                          atol=3e-2)
      self.assertAllClose(0.571912, loss.eval(), rtol=3e-2, atol=3e-2)
      predicted_labels = tf.cast(
          tf.greater_equal(prediction,
                           tf.ones_like(prediction) * 0.5), tf.float32)
      self.assertAllEqual([0, 1], predicted_labels.eval())


if __name__ == '__main__':
  googletest.main()
