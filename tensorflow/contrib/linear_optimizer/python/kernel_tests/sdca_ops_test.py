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


from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.contrib.linear_optimizer.python.ops.sdca_ops import SdcaModel
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.platform import googletest

_MAX_ITERATIONS = 100


def make_example_proto(feature_dict, target, value=1.0):
  e = tf.train.Example()
  features = e.features

  features.feature['target'].float_list.value.append(target)

  for key, values in feature_dict.items():
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


def make_dense_examples_dict(dense_feature_values, weights, labels):
  dense_feature_tensors = ([
      tf.convert_to_tensor(values,
                           dtype=tf.float32) for values in dense_feature_values
  ])
  return dict(sparse_features=[],
              dense_features=dense_feature_tensors,
              example_weights=weights,
              example_labels=labels,
              example_ids=['%d' % i for i in xrange(0, len(labels))])


def make_variable_dict(max_age, max_gender):
  # TODO(sibyl-toe9oF2e):  Figure out how to derive max_age & max_gender from
  # examples_dict.
  age_weights = tf.Variable(tf.zeros([max_age + 1], dtype=tf.float32))
  gender_weights = tf.Variable(tf.zeros([max_gender + 1], dtype=tf.float32))
  return dict(sparse_features_weights=[age_weights, gender_weights],
              dense_features_weights=[])


def make_dense_variable_dict(num_dense_features, num_examples):
  feature_weights = ([
      tf.Variable(tf.zeros([1],
                           dtype=tf.float32))
      for _ in xrange(0, num_dense_features)
  ])
  return dict(sparse_features_weights=[],
              dense_features_weights=feature_weights,
              dual=tf.Variable(tf.zeros(
                  [num_examples],
                  dtype=tf.float32)),
              primal_loss=tf.Variable(tf.zeros(
                  [],
                  dtype=tf.float64)))


def get_binary_predictions_for_logistic(predictions, cutoff=0.5):
  return tf.cast(
      tf.greater_equal(predictions, tf.ones_like(predictions) * cutoff),
      dtype=tf.int32)


def get_binary_predictions_for_hinge(predictions):
  return tf.cast(
      tf.greater_equal(predictions, tf.zeros_like(predictions)),
      dtype=tf.int32)


# Setup the single container shared across all tests. This is testing proper
# isolation across optimizers instantiated in each of the tests below.
CONTAINER = uuid.uuid4().hex


# Clear the shared container.
def tearDown():
  # TODO(sibyl-Mooth6ku): Proper cleanup of Containers when possible.
  pass


# TODO(sibyl-Mooth6ku): Add tests that exercise L1 and Shrinking.
# TODO(sibyl-vie3Poto): Refactor tests to avoid repetition of boilerplate code.
class SdcaOptimizerTest(TensorFlowTestCase):
  """Base SDCA optimizer test class for any loss type."""

  def _single_threaded_test_session(self):
    config = tf.ConfigProto(inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    return self.test_session(use_gpu=False, config=config)

  # The following tests, check that operations raise errors when certain
  # preconditions on the input data are not satisfied. These errors are raised
  # regardless of the loss type.
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
      options = dict(symmetric_l2_regularization=1,
                     symmetric_l1_regularization=0,
                     loss_type='logistic_loss')

      lr = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      self.assertAllClose([0.5, 0.5], lr.predictions(examples).eval())
      lr.minimize().run()
      self.assertAllClose([0.5, 0.5], lr.predictions(examples).eval())
      with self.assertRaisesOpError(
          'No examples found or all examples have zero weight.'):
        lr.approximate_duality_gap().eval()


class SdcaWithLogisticLossTest(SdcaOptimizerTest):
  """SDCA optimizer test class for logistic loss."""

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
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=1,
                     symmetric_l1_regularization=0,
                     loss_type='logistic_loss')

      lr = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      unregularized_loss = lr.unregularized_loss(examples)
      loss = lr.regularized_loss(examples)
      predictions = lr.predictions(examples)
      self.assertAllClose(0.693147, unregularized_loss.eval())
      self.assertAllClose(0.693147, loss.eval())
      train_op = lr.minimize()
      for _ in xrange(_MAX_ITERATIONS):
        train_op.run()
      # The high tolerance in unregularized_loss comparisons is due to the
      # fact that it's possible to trade off unregularized_loss vs.
      # regularization and still have a sum that is quite close to the
      # optimal regularized_loss value.  SDCA's duality gap only ensures that
      # the regularized_loss is within 0.01 of optimal.
      # 0.525457 is the optimal regularized_loss.
      # 0.411608 is the unregularized_loss at that optimum.
      self.assertAllClose(0.411608, unregularized_loss.eval(), atol=0.05)
      self.assertAllClose(0.525457, loss.eval(), atol=0.01)
      predicted_labels = get_binary_predictions_for_logistic(predictions)
      self.assertAllEqual([0, 1], predicted_labels.eval())
      self.assertAllClose(0.01,
                          lr.approximate_duality_gap().eval(),
                          rtol=1e-2,
                          atol=1e-2)

  def testSimpleNoL2(self):
    # Same as test above (so comments from above apply) but without an L2.
    # The algorithm should behave as if we have an L2 of 1 in optimization but
    # 0 in regularized_loss.
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
      options = dict(symmetric_l2_regularization=0,
                     symmetric_l1_regularization=0,
                     loss_type='logistic_loss')

      lr = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      unregularized_loss = lr.unregularized_loss(examples)
      loss = lr.regularized_loss(examples)
      predictions = lr.predictions(examples)
      self.assertAllClose(0.693147, unregularized_loss.eval())
      self.assertAllClose(0.693147, loss.eval())
      train_op = lr.minimize()
      for _ in xrange(_MAX_ITERATIONS):
        train_op.run()

      # There is neither L1 nor L2 loss, so regularized and unregularized losses
      # should be exactly the same.
      self.assertAllClose(0.40244, unregularized_loss.eval(), atol=0.01)
      self.assertAllClose(0.40244, loss.eval(), atol=0.01)
      predicted_labels = get_binary_predictions_for_logistic(predictions)
      self.assertAllEqual([0, 1], predicted_labels.eval())
      self.assertAllClose(0.01,
                          lr.approximate_duality_gap().eval(),
                          rtol=1e-2,
                          atol=1e-2)

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
      options = dict(symmetric_l2_regularization=1,
                     symmetric_l1_regularization=0,
                     loss_type='logistic_loss')

      lr = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      unregularized_loss = lr.unregularized_loss(examples)
      loss = lr.regularized_loss(examples)
      predictions = lr.predictions(examples)
      train_op = lr.minimize()
      for _ in xrange(_MAX_ITERATIONS):
        train_op.run()

      self.assertAllClose(0.411608, unregularized_loss.eval(), atol=0.05)
      self.assertAllClose(0.525457, loss.eval(), atol=0.01)
      predicted_labels = get_binary_predictions_for_logistic(predictions)
      self.assertAllClose([0, 1, 1, 1], predicted_labels.eval())
      self.assertAllClose(0.01,
                          lr.approximate_duality_gap().eval(),
                          rtol=1e-2,
                          atol=1e-2)

  def testFractionalExampleLabel(self):
    # Setup test data with 1 positive, and 1 mostly-negative example.
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0.1),
        make_example_proto(
            {'age': [1],
             'gender': [1]}, 1),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=1,
                     symmetric_l1_regularization=0,
                     loss_type='logistic_loss')

      lr = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      with self.assertRaisesOpError(
          'Only labels of 0.0 or 1.0 are supported right now.'):
        lr.minimize().run()

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
      options = dict(symmetric_l2_regularization=1,
                     symmetric_l1_regularization=0,
                     loss_type='logistic_loss')

      lr = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      unregularized_loss = lr.unregularized_loss(examples)
      loss = lr.regularized_loss(examples)
      predictions = lr.predictions(examples)
      train_op = lr.minimize()
      for _ in xrange(_MAX_ITERATIONS):
        train_op.run()

      self.assertAllClose(0.226487 + 0.102902,
                          unregularized_loss.eval(),
                          atol=0.08)
      self.assertAllClose(0.328394 + 0.131364, loss.eval(), atol=0.01)
      predicted_labels = get_binary_predictions_for_logistic(predictions)
      self.assertAllEqual([0, 0, 0, 1], predicted_labels.eval())
      self.assertAllClose(0.01,
                          lr.approximate_duality_gap().eval(),
                          rtol=1e-2,
                          atol=1e-2)

  def testImbalancedWithExampleWeights(self):
    # Setup test data with 1 positive, and 1 negative example.
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
      options = dict(symmetric_l2_regularization=1,
                     symmetric_l1_regularization=0,
                     loss_type='logistic_loss')

      lr = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      unregularized_loss = lr.unregularized_loss(examples)
      loss = lr.regularized_loss(examples)
      predictions = lr.predictions(examples)
      train_op = lr.minimize()
      for _ in xrange(_MAX_ITERATIONS):
        train_op.run()

      self.assertAllClose(0.284860, unregularized_loss.eval(), atol=0.08)
      self.assertAllClose(0.408044, loss.eval(), atol=0.012)
      predicted_labels = get_binary_predictions_for_logistic(predictions)
      self.assertAllEqual([0, 1], predicted_labels.eval())
      self.assertAllClose(0.01,
                          lr.approximate_duality_gap().eval(),
                          rtol=1e-2,
                          atol=1e-2)

  def testInstancesOfOneClassOnly(self):
    # Setup test data with 1 positive (ignored), and 1 negative example.
    example_protos = [
        make_example_proto(
            {'age': [0],
             'gender': [0]}, 0),
        make_example_proto(
            {'age': [1],
             'gender': [0]}, 1),  # Shares gender with the instance above.
    ]
    example_weights = [1.0, 0.0]  # Second example "omitted" from training.
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(symmetric_l2_regularization=1,
                     symmetric_l1_regularization=0,
                     loss_type='logistic_loss')

      lr = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      unregularized_loss = lr.unregularized_loss(examples)
      loss = lr.regularized_loss(examples)
      predictions = lr.predictions(examples)
      train_op = lr.minimize()
      for _ in xrange(_MAX_ITERATIONS):
        train_op.run()
      self.assertAllClose(0.411608, unregularized_loss.eval(), atol=0.05)
      self.assertAllClose(0.525457, loss.eval(), atol=0.01)
      predicted_labels = get_binary_predictions_for_logistic(predictions)
      self.assertAllEqual([0, 0], predicted_labels.eval())
      self.assertAllClose(0.01,
                          lr.approximate_duality_gap().eval(),
                          rtol=1e-2,
                          atol=1e-2)

  # TODO(katsiaspis): add a test for the case when examples at the end of an
  # epoch are repeated, since example id may be duplicated.

class SdcaWithLinearLossTest(SdcaOptimizerTest):
  """SDCA optimizer test class for linear (squared) loss."""

  def testSimple(self):
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
      options = dict(symmetric_l2_regularization=1,
                     symmetric_l1_regularization=0,
                     loss_type='squared_loss')

      lr = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      predictions = lr.predictions(examples)
      train_op = lr.minimize()
      for _ in xrange(_MAX_ITERATIONS):
        train_op.run()

      # Predictions should be 2/3 of label due to minimizing regularized loss:
      #   (label - 2 * weight)^2 / 2 + L2 * 2 * weight^2
      self.assertAllClose([-20.0 / 3.0, 28.0 / 3.0],
                          predictions.eval(),
                          rtol=0.005)
      self.assertAllClose(0.01,
                          lr.approximate_duality_gap().eval(),
                          rtol=1e-2,
                          atol=1e-2)

  def testL2Regularization(self):
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
      options = dict(symmetric_l2_regularization=16,
                     symmetric_l1_regularization=0,
                     loss_type='squared_loss')

      lr = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      predictions = lr.predictions(examples)

      train_op = lr.minimize()
      for _ in xrange(_MAX_ITERATIONS):
        train_op.run()

      # Predictions should be 1/5 of label due to minimizing regularized loss:
      #   (label - 2 * weight)^2 + L2 * 16 * weight^2
      optimal1 = -10.0 / 5.0
      optimal2 = 14.0 / 5.0
      self.assertAllClose(
          [optimal1, optimal1, optimal2, optimal2],
          predictions.eval(),
          rtol=0.01)

  def testL1Regularization(self):
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
      options = dict(symmetric_l2_regularization=1.0,
                     symmetric_l1_regularization=4.0,
                     loss_type='squared_loss')
      lr = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      prediction = lr.predictions(examples)
      loss = lr.regularized_loss(examples)

      train_op = lr.minimize()
      for _ in xrange(_MAX_ITERATIONS):
        train_op.run()

      # Predictions should be -4.0, 48/5 due to minimizing regularized loss:
      #   (label - 2 * weight)^2 / 2 + L2 * 2 * weight^2 + L1 * 4 * weight
      self.assertAllClose([-4.0, 20.0 / 3.0], prediction.eval(), rtol=0.08)

      # Loss should be the sum of the regularized loss value from above per
      # example after plugging in the optimal weights.
      self.assertAllClose(308.0 / 6.0, loss.eval(), atol=0.01)

  def testFeatureValues(self):
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
      options = dict(symmetric_l2_regularization=1,
                     symmetric_l1_regularization=0,
                     loss_type='squared_loss')

      lr = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      predictions = lr.predictions(examples)

      train_op = lr.minimize()
      for _ in xrange(_MAX_ITERATIONS):
        train_op.run()

      # Predictions should be 8/9 of label due to minimizing regularized loss:
      #   (label - 2 * 2 * weight)^2 / 2 + L2 * 2 * weight^2
      self.assertAllClose([-10.0 * 8 / 9, 14.0 * 8 / 9],
                          predictions.eval(),
                          rtol=0.07)

  def testDenseFeatures(self):
    with self._single_threaded_test_session():
      examples = make_dense_examples_dict(
          dense_feature_values=[[-2.0, 0.0], [0.0, 2.0]],
          weights=[1.0, 1.0],
          labels=[-10.0, 14.0])
      variables = make_dense_variable_dict(2, 2)
      options = dict(symmetric_l2_regularization=1,
                     symmetric_l1_regularization=0,
                     loss_type='squared_loss')
      lr = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      predictions = lr.predictions(examples)

      train_op = lr.minimize()
      for _ in xrange(_MAX_ITERATIONS):
        train_op.run()

      # Predictions should be 4/5 of label due to minimizing regularized loss:
      #   (label - 2 * weight)^2 / 2 + L2 * weight^2
      self.assertAllClose([-10.0 * 4 / 5, 14.0 * 4 / 5],
                          predictions.eval(),
                          rtol=0.01)

      loss = lr.regularized_loss(examples)
      self.assertAllClose(148.0 / 10.0, loss.eval(), atol=0.01)


class SdcaWithHingeLossTest(SdcaOptimizerTest):
  """SDCA optimizer test class for hinge loss."""

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
      options = dict(symmetric_l2_regularization=1.0,
                     symmetric_l1_regularization=0,
                     loss_type='hinge_loss')
      model = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()

      # Before minimization, the weights default to zero. There is no loss due
      # to regularization, only unregularized loss which is 0.5 * (1+1) = 1.0.
      predictions = model.predictions(examples)
      self.assertAllClose([0.0, 0.0], predictions.eval())
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      self.assertAllClose(1.0, unregularized_loss.eval())
      self.assertAllClose(1.0, regularized_loss.eval())

      # After minimization, the model separates perfectly the data points. There
      # are 4 sparse weights: 2 for age (say w1, w2) and 2 for gender (say w3
      # and w4). Solving the system w1 + w3 = 1.0, w2 + w4 = -1.0 and minimizing
      # wrt to \|\vec{w}\|_2, gives w1=w3=1/2 and w2=w4=-1/2. This gives 0.0
      # unregularized loss and 0.25 L2 loss.
      train_op = model.minimize()
      for _ in xrange(_MAX_ITERATIONS):
        train_op.run()

      binary_predictions = get_binary_predictions_for_hinge(predictions)
      self.assertAllEqual([-1.0, 1.0], predictions.eval())
      self.assertAllEqual([0, 1], binary_predictions.eval())
      self.assertAllClose(0.0, unregularized_loss.eval())
      self.assertAllClose(0.25, regularized_loss.eval(), atol=0.05)

  def testDenseFeaturesPerfectlySeparable(self):
    with self._single_threaded_test_session():
      examples = make_dense_examples_dict(
          dense_feature_values=[[1.0, 1.0], [1.0, -1.0]],
          weights=[1.0, 1.0],
          labels=[1.0, 0.0])
      variables = make_dense_variable_dict(2, 2)
      options = dict(symmetric_l2_regularization=1.0,
                     symmetric_l1_regularization=0,
                     loss_type='hinge_loss')
      model = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      predictions = model.predictions(examples)
      binary_predictions = get_binary_predictions_for_hinge(predictions)

      train_op = model.minimize()
      for _ in xrange(_MAX_ITERATIONS):
        train_op.run()

      self.assertAllClose([1.0, -1.0], predictions.eval(), atol=0.05)
      self.assertAllEqual([1, 0], binary_predictions.eval())

      # (1.0, 1.0) and (1.0, -1.0) are perfectly separable by x-axis (that is,
      # the SVM's functional margin >=1), so the unregularized loss is ~0.0.
      # There is only loss due to l2-regularization. For these datapoints, it
      # turns out that w_1~=0.0 and w_2~=1.0 which means that l2 loss is ~0.25.
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      self.assertAllClose(0.0, unregularized_loss.eval(), atol=0.02)
      self.assertAllClose(0.25, regularized_loss.eval(), atol=0.02)

  def testDenseFeaturesSeparableWithinMargins(self):
    with self._single_threaded_test_session():
      examples = make_dense_examples_dict(
          dense_feature_values=[[1.0, 1.0], [0.5, -0.5]],
          weights=[1.0, 1.0],
          labels=[1.0, 0.0])
      variables = make_dense_variable_dict(2, 2)
      options = dict(symmetric_l2_regularization=1.0,
                     symmetric_l1_regularization=0,
                     loss_type='hinge_loss')
      model = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      predictions = model.predictions(examples)
      binary_predictions = get_binary_predictions_for_hinge(predictions)

      train_op = model.minimize()
      for _ in xrange(_MAX_ITERATIONS):
        train_op.run()

      # (1.0, 0.5) and (1.0, -0.5) are separable by x-axis but the datapoints
      # are within the margins so there is unregularized loss (1/2 per example).
      # For these datapoints, optimal weights are w_1~=0.0 and w_2~=1.0 which
      # gives an L2 loss of ~0.25.
      self.assertAllClose([0.5, -0.5], predictions.eval(), rtol=0.05)
      self.assertAllEqual([1, 0], binary_predictions.eval())
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      self.assertAllClose(0.5, unregularized_loss.eval(), atol=0.02)
      self.assertAllClose(0.75, regularized_loss.eval(), atol=0.02)

  def testDenseFeaturesWeightedExamples(self):
    with self._single_threaded_test_session():
      examples = make_dense_examples_dict(
          dense_feature_values=[[1.0, 1.0], [0.5, -0.5]],
          weights=[3.0, 1.0],
          labels=[1.0, 0.0])
      variables = make_dense_variable_dict(2, 2)
      options = dict(symmetric_l2_regularization=1.0,
                     symmetric_l1_regularization=0,
                     loss_type='hinge_loss')
      model = SdcaModel(CONTAINER, examples, variables, options)
      tf.initialize_all_variables().run()
      predictions = model.predictions(examples)
      binary_predictions = get_binary_predictions_for_hinge(predictions)
      train_op = model.minimize()
      for _ in xrange(_MAX_ITERATIONS):
        train_op.run()

      # Point (1.0, 0.5) has higher weight than (1.0, -0.5) so the model will
      # try to increase the margin from (1.0, 0.5). Due to regularization,
      # (1.0, -0.5) will be within the margin. For these points and example
      # weights, the optimal weights are w_1~=0.4 and w_2~=1.2 which give an L2
      # loss of 0.5 * 0.25 * 0.25 * 1.6 = 0.2. The binary predictions will be
      # correct, but the boundary will be much closer to the 2nd point than the
      # first one.
      self.assertAllClose([1.0, -0.2], predictions.eval(), atol=0.05)
      self.assertAllEqual([1, 0], binary_predictions.eval())
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      self.assertAllClose(0.2, unregularized_loss.eval(), atol=0.02)
      self.assertAllClose(0.4, regularized_loss.eval(), atol=0.02)

if __name__ == '__main__':
  googletest.main()
